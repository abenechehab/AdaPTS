from typing import Optional, Union, Any
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection

import torch
import torch.nn as nn


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        A custom scikit-learn transformer that performs no operation on the data.
        This is useful for the vICL variant of DICL.

        Methods:
            fit(
                input_array: NDArray, y: Optional[NDArray] = None
            ) -> IdentityTransformer:
                Fits the transformer (no-op in this case), returning the instance
                itself.

            transform(input_array: NDArray, y: Optional[NDArray] = None) -> NDArray:
                Returns the input data without modification.

            inverse_transform(
                input_array: NDArray, y: Optional[NDArray] = None) -> NDArray:
                    Returns the input data without modification.
        """
        pass

    def fit(self, input_array: NDArray, y: Optional[NDArray] = None):
        return self

    def transform(self, input_array: NDArray, y: Optional[NDArray] = None) -> NDArray:
        return input_array * 1

    def inverse_transform(
        self, input_array: NDArray, y: Optional[NDArray] = None
    ) -> NDArray:
        return input_array * 1


class MultichannelProjector:
    """
    A class used to project multichannel data using various dimensionality reduction
    techniques.
    Attributes
    ----------
    new_num_channels : int
        The number of channels after projection.
    patch_window_size : int, optional
        The size of the patch window. If None, defaults to 1.
    patch_window_size_ : int
        The effective patch window size used internally.
    base_projector : str or object, optional
        The base projector to use. Can be 'pca', 'svd', 'rand', or a custom projector
        object.
    base_projector_ : object
        The instantiated base projector used for dimensionality reduction.
    Methods
    -------
    fit(X)
        Fits the base projector to the input data X.
    transform(X)
        Transforms the input data X using the fitted base projector.
    """

    def __init__(
        self,
        num_channels: int,
        new_num_channels: int,
        patch_window_size: Optional[int] = None,
        base_projector: Optional[Union[str, Any]] = None,
        device: str = "cpu",
    ):
        # init dimensions
        self.num_channels = num_channels
        self.new_num_channels = new_num_channels
        self.patch_window_size = patch_window_size
        self.patch_window_size_ = 1 if patch_window_size is None else patch_window_size
        # init base projector
        self.base_projector = base_projector
        n_components = self.patch_window_size_ * new_num_channels
        if base_projector is None:
            self.base_projector_ = IdentityTransformer()
        elif base_projector == "pca":
            self.base_projector_ = PCA(n_components=n_components)
        elif base_projector == "svd":
            self.base_projector_ = TruncatedSVD(n_components=n_components)
        elif base_projector == "rand":
            self.base_projector_ = SparseRandomProjection(n_components=n_components)
        elif base_projector == "simpleAE":
            self.base_projector_ = SimpleAutoEncoder(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
            ).to(torch.device(device))
        elif base_projector == "linearAE":
            self.base_projector_ = LinearAutoEncoder(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
            ).to(torch.device(device))
        # you can give your own base_projector with fit() and transform() methods, and
        # it should have the argument `n_components`.
        else:
            self.base_projector_ = base_projector(n_components=n_components)

    def fit(self, X, y: Optional[NDArray] = None):
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )
        return self.base_projector_.fit(X_2d)

    def transform(self, X, y: Optional[NDArray] = None):
        """
        Apply the PCA transform on the input data.
        """
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )

        X_transformed = self.base_projector_.transform(X_2d)
        X_transformed = X_transformed.reshape(
            [num_samples, seq_len, self.new_num_channels]
        )
        return np.swapaxes(X_transformed, 1, 2)

    def transform_torch(self, X, y: Optional[NDArray] = None):
        """
        Apply the PCA transform on the input data.
        """

        assert hasattr(self.base_projector_, "transform_torch"), "Base projector must"
        " implement transform_torch method"

        X_transposed = torch.transpose(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )

        X_transformed = self.base_projector_.transform_torch(X_2d)
        X_transformed = X_transformed.reshape(
            [num_samples, seq_len, self.new_num_channels]
        )
        return torch.transpose(X_transformed, 1, 2)

    def inverse_transform(self, X, y: Optional[NDArray] = None):
        """
        Apply the base_projector_ inverse transform on the input data.
        """
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.new_num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )

        X_inverse_transformed = self.base_projector_.inverse_transform(X_2d)
        X_inverse_transformed = X_inverse_transformed.reshape(
            [num_samples, seq_len, self.num_channels]
        )
        return np.swapaxes(X_inverse_transformed, 1, 2)

    def inverse_transform_torch(self, X, y: Optional[NDArray] = None):
        """
        Apply the base_projector_ inverse transform on the input data.
        """
        assert hasattr(self.base_projector_, "inverse_transform_torch"), "Base "
        "projector must implement inverse_transform_torch method"

        X_transposed = torch.transpose(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        assert num_channels == self.new_num_channels, "Number of channels must match."
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(
            num_samples * num_patches, self.patch_window_size_ * num_channels
        )

        X_inverse_transformed = self.base_projector_.inverse_transform_torch(X_2d)
        X_inverse_transformed = X_inverse_transformed.reshape(
            [num_samples, seq_len, self.num_channels]
        )
        return torch.transpose(X_inverse_transformed, 1, 2)


class LinearAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        device: str = "cpu",
    ):
        """
        Initialize AutoEncoder for feature space projection.

        Args:
        input_dim: Input dimension
        n_components: Desired output dimension (latent space)
        num_layers: Number of layers in encoder and decoder
        hidden_dim: Number of neurons in hidden layers
        """
        super().__init__()

        self.device = torch.device(device)

        # Build encoder layers
        self.encoder = nn.Sequential()
        self.encoder.add_module("layer0", nn.Linear(input_dim, n_components))

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, input_dim))

    def forward(self, x):
        """Forward pass through autoencoder"""
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X,
        y=None,
        train_proportion=0.8,
        n_epochs=100,
        early_stopping_patience=10,
        learning_rate=1e-3,
        verbose=1,
    ):
        """Compatibility with sklearn interface"""
        # Move model to specified device
        self.to(torch.device(self.device))

        # Convert data to tensor
        X_tensor = torch.FloatTensor(X).to(torch.device(self.device))

        # Split data into train and validation (80-20 split)
        train_size = int(train_proportion * len(X_tensor))
        train_data = X_tensor[:train_size]
        val_data = X_tensor[train_size:]

        # Define optimizer, scheduler and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()

        # Create DataLoader for training in batches
        train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        # Training with batches
        best_val_loss = float("inf")
        patience_counter = 0

        # Train for 100 epochs
        for _ in tqdm(range(n_epochs), disable=not verbose, desc="Training Epochs"):
            # Training phase
            self.train()
            epoch_train_loss = 0
            for batch_x, batch_y in train_loader:
                output = self(batch_x)
                train_loss = criterion(output, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_train_loss += train_loss.item()

            # Validation phase
            self.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    val_output = self(batch_x)
                    val_loss = criterion(val_output, batch_y)
                    epoch_val_loss += val_loss.item()

            # Update scheduler
            scheduler.step(epoch_val_loss)

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

        return self

    def transform(self, X):
        """Project data to latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.encoder(X_tensor).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        return self.encoder(X)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        return self.decoder(X)


class SimpleAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        num_layers: int = 2,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        """
        Initialize AutoEncoder for feature space projection.

        Args:
        input_dim: Input dimension
        n_components: Desired output dimension (latent space)
        num_layers: Number of layers in encoder and decoder
        hidden_dim: Number of neurons in hidden layers
        """
        super().__init__()

        self.device = torch.device(device)

        # Build encoder layers
        self.encoder = nn.Sequential()
        self.encoder.add_module("layer0", nn.Linear(input_dim, hidden_dim))
        self.encoder.add_module("layer0-bn", nn.BatchNorm1d(int(hidden_dim)))
        self.encoder.add_module("layer0-act", nn.ReLU())
        for i in range(1, num_layers):
            self.encoder.add_module(f"layer{i}", nn.Linear(hidden_dim, hidden_dim))
            self.encoder.add_module(f"layer{i}-bn", nn.BatchNorm1d(int(hidden_dim)))
            self.encoder.add_module(f"layer{i}-act", nn.ReLU())
        self.encoder.add_module(
            f"layer{num_layers}", nn.Linear(hidden_dim, n_components)
        )
        self.encoder.add_module(f"layer{num_layers}-act", nn.ReLU())

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, hidden_dim))
        self.decoder.add_module("layer0-bn", nn.BatchNorm1d(int(hidden_dim)))
        self.decoder.add_module("layer0-act", nn.ReLU())
        for i in range(1, num_layers):
            self.decoder.add_module(f"layer{i}", nn.Linear(hidden_dim, hidden_dim))
            self.decoder.add_module(f"layer{i}-bn", nn.BatchNorm1d(int(hidden_dim)))
            self.decoder.add_module(f"layer{i}-act", nn.ReLU())
        self.decoder.add_module(f"layer{num_layers}", nn.Linear(hidden_dim, input_dim))
        self.decoder.add_module(f"layer{num_layers}-act", nn.ReLU())

    def forward(self, x):
        """Forward pass through autoencoder"""
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X,
        y=None,
        train_proportion=0.8,
        n_epochs=100,
        early_stopping_patience=10,
        learning_rate=1e-3,
        verbose=1,
    ):
        """Compatibility with sklearn interface"""
        # Move model to specified device
        self.to(torch.device(self.device))

        # Convert data to tensor
        X_tensor = torch.FloatTensor(X).to(torch.device(self.device))

        # Split data into train and validation (80-20 split)
        train_size = int(train_proportion * len(X_tensor))
        train_data = X_tensor[:train_size]
        val_data = X_tensor[train_size:]

        # Define optimizer, scheduler and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()

        # Create DataLoader for training in batches
        train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        # Training with batches
        best_val_loss = float("inf")
        patience_counter = 0

        # Train for 100 epochs
        for _ in tqdm(range(n_epochs), disable=not verbose, desc="Training Epochs"):
            # Training phase
            self.train()
            epoch_train_loss = 0
            for batch_x, batch_y in train_loader:
                output = self(batch_x)
                train_loss = criterion(output, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_train_loss += train_loss.item()

            # Validation phase
            self.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    val_output = self(batch_x)
                    val_loss = criterion(val_output, batch_y)
                    epoch_val_loss += val_loss.item()

            # Update scheduler
            scheduler.step(epoch_val_loss)

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

        return self

    def transform(self, X):
        """Project data to latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.encoder(X_tensor).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        return self.encoder(X)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        return self.decoder(X)
