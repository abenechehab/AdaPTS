from typing import Optional, Union, Any
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection

import torch
import torch.nn as nn

from dicl.utils.preprocessing import RevIN


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
        use_revin: bool = False,
        context_length: int = 512,
        forecast_horizon: int = 96,
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
        elif base_projector == "linearAE":
            self.base_projector_ = LinearAutoEncoder(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "dropoutLinearAE":
            self.base_projector_ = DropoutLinearAutoEncoder(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "linearDecoder":
            self.base_projector_ = LinearDecoder(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "linearEncoder":
            self.base_projector_ = LinearEncoder(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "simpleAE":
            self.base_projector_ = SimpleAutoEncoder(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
            ).to(torch.device(device))
        elif base_projector == "VAE":
            self.base_projector_ = betaVAE(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "linearVAE":
            self.base_projector_ = betaLinearVAE(
                n_components=n_components,
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "flow":
            self.base_projector_ = NormalizingFlow(
                input_dim=self.num_channels,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "AEflow":
            self.base_projector_ = AENormalizingFlow(
                input_dim=self.num_channels,
                n_components=n_components,
                device=device,
                use_revin=use_revin,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
            ).to(torch.device(device))
        elif base_projector == "revin":
            self.base_projector_ = JustRevIn(
                num_features=self.num_channels,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
                device=device,
            ).to(torch.device(device))
        # you can give your own base_projector with fit() and transform() methods, and
        # it should have the argument `n_components`.
        else:
            self.base_projector_ = base_projector

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
        context_length: int,
        forecast_horizon: int,
        device: str = "cpu",
        use_revin: bool = False,
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

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim
        self.n_components = n_components

        self.device = torch.device(device)

        # Build encoder layers
        self.encoder = nn.Sequential()
        self.encoder.add_module("layer0", nn.Linear(input_dim, n_components))

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, input_dim))

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def forward(self, x):
        """Forward pass through autoencoder"""
        if self.use_revin:
            revin_input = x.reshape(-1, self.context_length, self.input_dim)
            after_encoding = self.encoder(
                self.revin(revin_input, mode="norm").reshape(-1, self.n_components)
            )
            after_decoding = self.decoder(after_encoding)
            reverse_revin_input = after_decoding.reshape(
                -1, self.context_length, self.input_dim
            )
            return self.revin(
                reverse_revin_input,
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X,
        y=None,
        train_proportion=0.8,
        n_epochs=300,
        early_stopping_patience=10,
        learning_rate=1e-3,
        verbose=1,
    ):
        """Compatibility with sklearn interface"""
        # Move model to specified device
        self.to(torch.device(self.device))

        # Convert data to tensor
        X_tensor = torch.FloatTensor(X).to(torch.device(self.device))
        X_tensor = X_tensor.reshape(-1, self.context_length, self.input_dim)

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
                batch_x, batch_y = batch_x.reshape(-1, self.input_dim), batch_y.reshape(
                    -1, self.input_dim
                )
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
                    batch_x, batch_y = (
                        batch_x.reshape(-1, self.input_dim),
                        batch_y.reshape(-1, self.input_dim),
                    )
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
            if self.use_revin:
                X_tensor = self.revin(
                    X_tensor.reshape(-1, self.context_length, self.input_dim),
                    mode="norm",
                ).reshape(-1, self.input_dim)
            return self.encoder(X_tensor).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        if self.use_revin:
            X = self.revin(
                X.reshape(-1, self.context_length, self.input_dim), mode="norm"
            ).reshape(-1, self.input_dim)
        return self.encoder(X)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                return (
                    self.revin(
                        self.decoder(X_tensor).reshape(
                            -1, self.forecast_horizon, self.input_dim
                        ),
                        mode="denorm",
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1, self.input_dim)
                )
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        if self.use_revin:
            return self.revin(
                self.decoder(X).reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(X)

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        X_batch = X_batch.to(self.device)
        X_reconstructed = self(X_batch)
        return nn.MSELoss()(X_reconstructed, X_batch)


class DropoutLinearAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        context_length: int,
        forecast_horizon: int,
        device: str = "cpu",
        use_revin: bool = False,
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

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim
        self.n_components = n_components

        self.device = torch.device(device)

        # Build encoder layers
        self.encoder = nn.Sequential()
        self.encoder.add_module("layer0", nn.Linear(input_dim, n_components))
        self.encoder.add_module("layer0-dropout", nn.Dropout(p=0.1))

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, input_dim))

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def forward(self, x):
        """Forward pass through autoencoder"""
        if self.use_revin:
            revin_input = x.reshape(-1, self.context_length, self.input_dim)
            after_encoding = self.encoder(
                self.revin(revin_input, mode="norm").reshape(-1, self.n_components)
            )
            after_decoding = self.decoder(after_encoding)
            reverse_revin_input = after_decoding.reshape(
                -1, self.context_length, self.input_dim
            )
            return self.revin(
                reverse_revin_input,
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X,
        y=None,
        train_proportion=0.8,
        n_epochs=300,
        early_stopping_patience=10,
        learning_rate=1e-3,
        verbose=1,
    ):
        """Compatibility with sklearn interface"""
        # Move model to specified device
        self.to(torch.device(self.device))

        # Convert data to tensor
        X_tensor = torch.FloatTensor(X).to(torch.device(self.device))
        X_tensor = X_tensor.reshape(-1, self.context_length, self.input_dim)

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
                batch_x, batch_y = (
                    batch_x.reshape(-1, self.input_dim),
                    batch_y.reshape(-1, self.input_dim),
                )
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
                    batch_x, batch_y = (
                        batch_x.reshape(-1, self.input_dim),
                        batch_y.reshape(-1, self.input_dim),
                    )
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
            if self.use_revin:
                X_tensor = self.revin(
                    X_tensor.reshape(-1, self.context_length, self.input_dim),
                    mode="norm",
                ).reshape(-1, self.input_dim)
            return self.encoder(X_tensor).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        if self.use_revin:
            X = self.revin(
                X.reshape(-1, self.context_length, self.input_dim), mode="norm"
            ).reshape(-1, self.input_dim)
        return self.encoder(X)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                return (
                    self.revin(
                        self.decoder(X_tensor).reshape(
                            -1, self.forecast_horizon, self.input_dim
                        ),
                        mode="denorm",
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1, self.input_dim)
                )
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        if self.use_revin:
            return self.revin(
                self.decoder(X).reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(X)

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        X_batch = X_batch.to(self.device)
        X_reconstructed = self(X_batch)
        return nn.MSELoss()(X_reconstructed, X_batch)


class LinearDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        context_length: int,
        forecast_horizon: int,
        device: str = "cpu",
        use_revin: bool = False,
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

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim
        self.n_components = n_components

        self.device = torch.device(device)

        # Build encoder layers
        self.encoder = nn.Identity()

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, input_dim))

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def forward(self, x):
        """Forward pass through autoencoder"""
        if self.use_revin:
            revin_input = x.reshape(-1, self.context_length, self.input_dim)
            after_encoding = self.encoder(
                self.revin(revin_input, mode="norm").reshape(-1, self.n_components)
            )
            after_decoding = self.decoder(after_encoding)
            reverse_revin_input = after_decoding.reshape(
                -1, self.context_length, self.input_dim
            )
            return self.revin(
                reverse_revin_input,
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X,
        y=None,
        train_proportion=0.8,
        n_epochs=300,
        early_stopping_patience=10,
        learning_rate=1e-3,
        verbose=1,
    ):
        """Compatibility with sklearn interface"""
        # Move model to specified device
        self.to(torch.device(self.device))

        # Convert data to tensor
        X_tensor = torch.FloatTensor(X).to(torch.device(self.device))
        X_tensor = X_tensor.reshape(-1, self.context_length, self.input_dim)

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
                batch_x, batch_y = (
                    batch_x.reshape(-1, self.input_dim),
                    batch_y.reshape(-1, self.input_dim),
                )
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
                    batch_x, batch_y = (
                        batch_x.reshape(-1, self.input_dim),
                        batch_y.reshape(-1, self.input_dim),
                    )
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
            if self.use_revin:
                X_tensor = self.revin(
                    X_tensor.reshape(-1, self.context_length, self.input_dim),
                    mode="norm",
                ).reshape(-1, self.input_dim)
            return self.encoder(X_tensor).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        if self.use_revin:
            X = self.revin(
                X.reshape(-1, self.context_length, self.input_dim), mode="norm"
            ).reshape(-1, self.input_dim)
        return self.encoder(X)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                return (
                    self.revin(
                        self.decoder(X_tensor).reshape(
                            -1, self.forecast_horizon, self.input_dim
                        ),
                        mode="denorm",
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1, self.input_dim)
                )
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        if self.use_revin:
            return self.revin(
                self.decoder(X).reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(X)

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        X_batch = X_batch.to(self.device)
        X_reconstructed = self(X_batch)
        return nn.MSELoss()(X_reconstructed, X_batch)


class LinearEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        context_length: int,
        forecast_horizon: int,
        device: str = "cpu",
        use_revin: bool = False,
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

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim
        self.n_components = n_components

        self.device = torch.device(device)

        # Build encoder layers
        self.encoder = nn.Sequential()
        self.encoder.add_module("layer0", nn.Linear(input_dim, n_components))

        # Build decoder layers
        self.decoder = nn.Identity()

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def forward(self, x):
        """Forward pass through autoencoder"""
        if self.use_revin:
            revin_input = x.reshape(-1, self.context_length, self.input_dim)
            after_encoding = self.encoder(
                self.revin(revin_input, mode="norm").reshape(-1, self.n_components)
            )
            after_decoding = self.decoder(after_encoding)
            reverse_revin_input = after_decoding.reshape(
                -1, self.context_length, self.input_dim
            )
            return self.revin(
                reverse_revin_input,
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X,
        y=None,
        train_proportion=0.8,
        n_epochs=300,
        early_stopping_patience=10,
        learning_rate=1e-3,
        verbose=1,
    ):
        """Compatibility with sklearn interface"""
        # Move model to specified device
        self.to(torch.device(self.device))

        # Convert data to tensor
        X_tensor = torch.FloatTensor(X).to(torch.device(self.device))
        X_tensor = X_tensor.reshape(-1, self.context_length, self.input_dim)

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
                batch_x, batch_y = (
                    batch_x.reshape(-1, self.input_dim),
                    batch_y.reshape(-1, self.input_dim),
                )
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
                    batch_x, batch_y = (
                        batch_x.reshape(-1, self.input_dim),
                        batch_y.reshape(-1, self.input_dim),
                    )
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
            if self.use_revin:
                X_tensor = self.revin(
                    X_tensor.reshape(-1, self.context_length, self.input_dim),
                    mode="norm",
                ).reshape(-1, self.input_dim)
            return self.encoder(X_tensor).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        if self.use_revin:
            X = self.revin(
                X.reshape(-1, self.context_length, self.input_dim), mode="norm"
            ).reshape(-1, self.input_dim)
        return self.encoder(X)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                return (
                    self.revin(
                        self.decoder(X_tensor).reshape(
                            -1, self.forecast_horizon, self.input_dim
                        ),
                        mode="denorm",
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1, self.input_dim)
                )
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        if self.use_revin:
            return self.revin(
                self.decoder(X).reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(X)

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        X_batch = X_batch.to(self.device)
        X_reconstructed = self(X_batch)
        return nn.MSELoss()(X_reconstructed, X_batch)


class SimpleAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        num_layers: int = 2,
        hidden_dim: int = 128,
        device: str = "cpu",
        use_revin: bool = False,
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
        self.encoder.add_module("layer0-act", nn.LeakyReLU())
        for i in range(1, num_layers):
            self.encoder.add_module(f"layer{i}", nn.Linear(hidden_dim, hidden_dim))
            self.encoder.add_module(f"layer{i}-bn", nn.BatchNorm1d(int(hidden_dim)))
            self.encoder.add_module(f"layer{i}-act", nn.LeakyReLU())
        self.encoder.add_module(
            f"layer{num_layers+1}", nn.Linear(hidden_dim, n_components)
        )
        self.encoder.add_module(f"layer{num_layers+1}-act", nn.LeakyReLU())

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, hidden_dim))
        self.decoder.add_module("layer0-bn", nn.BatchNorm1d(int(hidden_dim)))
        self.decoder.add_module("layer0-act", nn.LeakyReLU())
        for i in range(1, num_layers):
            self.decoder.add_module(f"layer{i}", nn.Linear(hidden_dim, hidden_dim))
            self.decoder.add_module(f"layer{i}-bn", nn.BatchNorm1d(int(hidden_dim)))
            self.decoder.add_module(f"layer{i}-act", nn.LeakyReLU())
        self.decoder.add_module(
            f"layer{num_layers+1}", nn.Linear(hidden_dim, input_dim)
        )
        self.decoder.add_module(f"layer{num_layers+1}-act", nn.Tanh())

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def forward(self, x):
        """Forward pass through autoencoder"""
        if self.use_revin:
            return self.revin(
                self.decoder(self.encoder(self.revin(x, mode="norm"))), mode="denorm"
            )
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
            if self.use_revin:
                X_tensor = self.revin(X_tensor, mode="norm")
            return self.encoder(X_tensor).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        if self.use_revin:
            X = self.revin(X, mode="norm")
        return self.encoder(X)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        return self.decoder(X)

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        X_reconstructed = self(X_batch)
        return nn.MSELoss()(X_reconstructed, X_batch)


class betaVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        context_length: int,
        forecast_horizon: int,
        num_layers: int = 1,
        hidden_dim: int = 64,
        beta: float = 0.5,
        device: str = "cpu",
        use_revin: bool = False,
    ):
        """
        Initialize Variational AutoEncoder for feature space projection.

        Args:
        input_dim: Input dimension
        n_components: Desired output dimension (latent space)
        num_layers: Number of layers in encoder and decoder
        hidden_dim: Number of neurons in hidden layers
        """
        super().__init__()

        self.device = torch.device(device)
        self.beta = beta
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.n_components = n_components
        self.input_dim = input_dim

        # Build encoder layers
        self.encoder = nn.Sequential()
        self.encoder.add_module("layer0", nn.Linear(input_dim, hidden_dim))
        self.encoder.add_module("layer0-bn", nn.BatchNorm1d(int(hidden_dim)))
        self.encoder.add_module("layer0-act", nn.LeakyReLU())
        for i in range(1, num_layers):
            self.encoder.add_module(f"layer{i}", nn.Linear(hidden_dim, hidden_dim))
            self.encoder.add_module(f"layer{i}-bn", nn.BatchNorm1d(int(hidden_dim)))
            self.encoder.add_module(f"layer{i}-act", nn.LeakyReLU())

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, hidden_dim))
        self.decoder.add_module("layer0-bn", nn.BatchNorm1d(int(hidden_dim)))
        self.decoder.add_module("layer0-act", nn.LeakyReLU())
        for i in range(1, num_layers):
            self.decoder.add_module(f"layer{i}", nn.Linear(hidden_dim, hidden_dim))
            self.decoder.add_module(f"layer{i}-bn", nn.BatchNorm1d(int(hidden_dim)))
            self.decoder.add_module(f"layer{i}-act", nn.LeakyReLU())
        self.decoder.add_module(
            f"layer{num_layers+1}", nn.Linear(hidden_dim, input_dim)
        )
        self.decoder.add_module(f"layer{num_layers+1}-act", nn.Tanh())

        # Build latent space layers
        self.latent_mu = nn.Linear(hidden_dim, n_components)
        self.latent_logvar = nn.Linear(hidden_dim, n_components)

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through autoencoder"""
        if self.use_revin:
            revin_input = x.reshape(-1, self.context_length, self.input_dim)
            after_encoding = self.encoder(
                self.revin(revin_input, mode="norm").reshape(-1, self.input_dim)
            )
            mu, logvar = (
                self.latent_mu(after_encoding),
                self.latent_logvar(after_encoding),
            )
            z = self.reparameterize(mu, logvar)
            after_decoding = self.decoder(z)
            reverse_revin_input = after_decoding.reshape(
                -1, self.context_length, self.input_dim
            )
            return self.revin(
                reverse_revin_input,
                mode="denorm",
            ).reshape(-1, self.input_dim)
        encoding = self.encoder(x)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

    def transform(self, X):
        """Project data to latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                X_tensor = self.revin(
                    X_tensor.reshape(-1, self.context_length, self.input_dim),
                    mode="norm",
                ).reshape(-1, self.input_dim)
            encoding = self.encoder(X_tensor)
            mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
            return self.reparameterize(mu, logvar).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        if self.use_revin:
            X = self.revin(
                X.reshape(-1, self.context_length, self.input_dim), mode="norm"
            ).reshape(-1, self.input_dim)
        encoding = self.encoder(X)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        return self.reparameterize(mu, logvar)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                return (
                    self.revin(
                        self.decoder(X_tensor).reshape(
                            -1, self.forecast_horizon, self.input_dim
                        ),
                        mode="denorm",
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1, self.input_dim)
                )
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        if self.use_revin:
            return self.revin(
                self.decoder(X).reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(X)

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        X_batch = X_batch.to(self.device)
        if self.use_revin:
            X_batch = self.revin(
                X_batch.reshape(-1, self.context_length, self.input_dim),
                mode="norm",
            ).reshape(-1, self.n_components)
        encoding = self.encoder(X_batch)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        z = self.reparameterize(mu, logvar)
        after_decoding = self.decoder(z)
        if self.use_revin:
            after_decoding = self.revin(
                after_decoding.reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        reconstruction_loss = nn.MSELoss()(after_decoding, X_batch)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.beta * kl_loss

    def kl_loss(self, X_batch):
        """Compute KL loss"""
        X_batch = X_batch.to(self.device)
        if self.use_revin:
            X_batch = self.revin(
                X_batch.reshape(-1, self.context_length, self.input_dim),
                mode="norm",
            ).reshape(-1, self.n_components)
        encoding = self.encoder(X_batch)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return self.beta * kl_loss

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


class betaLinearVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        context_length: int,
        forecast_horizon: int,
        num_layers: int = 1,
        hidden_dim: int = 64,
        beta: float = 0.5,
        device: str = "cpu",
        use_revin: bool = False,
    ):
        """
        Initialize Variational AutoEncoder for feature space projection.

        Args:
        input_dim: Input dimension
        n_components: Desired output dimension (latent space)
        num_layers: Number of layers in encoder and decoder
        hidden_dim: Number of neurons in hidden layers
        """
        super().__init__()

        self.device = torch.device(device)
        self.beta = beta
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.n_components = n_components
        self.input_dim = input_dim

        # Build encoder layers
        self.encoder = nn.Identity()

        # Build decoder layers
        self.decoder = nn.Sequential()
        self.decoder.add_module("layer0", nn.Linear(n_components, input_dim))

        # Build latent space layers
        self.latent_mu = nn.Linear(input_dim, n_components)
        self.latent_logvar = nn.Linear(input_dim, n_components)

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through autoencoder"""
        if self.use_revin:
            revin_input = x.reshape(-1, self.context_length, self.input_dim)
            after_encoding = self.encoder(
                self.revin(revin_input, mode="norm").reshape(-1, self.input_dim)
            )
            mu, logvar = (
                self.latent_mu(after_encoding),
                self.latent_logvar(after_encoding),
            )
            z = self.reparameterize(mu, logvar)
            after_decoding = self.decoder(z)
            reverse_revin_input = after_decoding.reshape(
                -1, self.context_length, self.input_dim
            )
            return self.revin(
                reverse_revin_input,
                mode="denorm",
            ).reshape(-1, self.input_dim)
        encoding = self.encoder(x)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

    def transform(self, X):
        """Project data to latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                X_tensor = self.revin(
                    X_tensor.reshape(-1, self.context_length, self.input_dim),
                    mode="norm",
                ).reshape(-1, self.input_dim)
            encoding = self.encoder(X_tensor)
            mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
            return self.reparameterize(mu, logvar).cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space"""
        if self.use_revin:
            X = self.revin(
                X.reshape(-1, self.context_length, self.input_dim), mode="norm"
            ).reshape(-1, self.input_dim)
        encoding = self.encoder(X)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        return self.reparameterize(mu, logvar)

    def inverse_transform(self, X):
        """Reconstruct from latent space"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if self.use_revin:
                return (
                    self.revin(
                        self.decoder(X_tensor).reshape(
                            -1, self.forecast_horizon, self.input_dim
                        ),
                        mode="denorm",
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1, self.input_dim)
                )
            return self.decoder(X_tensor).cpu().detach().numpy()

    def inverse_transform_torch(self, X):
        """Reconstruct from latent space"""
        if self.use_revin:
            return self.revin(
                self.decoder(X).reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        return self.decoder(X)

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        X_batch = X_batch.to(self.device)
        if self.use_revin:
            X_batch = self.revin(
                X_batch.reshape(-1, self.context_length, self.input_dim),
                mode="norm",
            ).reshape(-1, self.n_components)
        encoding = self.encoder(X_batch)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        z = self.reparameterize(mu, logvar)
        after_decoding = self.decoder(z)
        if self.use_revin:
            after_decoding = self.revin(
                after_decoding.reshape(-1, self.forecast_horizon, self.input_dim),
                mode="denorm",
            ).reshape(-1, self.input_dim)
        reconstruction_loss = nn.MSELoss()(after_decoding, X_batch)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.beta * kl_loss

    def kl_loss(self, X_batch):
        """Compute KL loss"""
        X_batch = X_batch.to(self.device)
        if self.use_revin:
            X_batch = self.revin(
                X_batch.reshape(-1, self.context_length, self.input_dim),
                mode="norm",
            ).reshape(-1, self.n_components)
        encoding = self.encoder(X_batch)
        mu, logvar = self.latent_mu(encoding), self.latent_logvar(encoding)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return self.beta * kl_loss

    def fit(
        self,
        X,
        y=None,
        train_proportion=0.8,
        n_epochs=300,
        early_stopping_patience=10,
        learning_rate=1e-3,
        verbose=1,
    ):
        """Compatibility with sklearn interface"""
        # Move model to specified device
        self.to(torch.device(self.device))

        # Convert data to tensor
        X_tensor = torch.FloatTensor(X).to(torch.device(self.device))
        X_tensor = X_tensor.reshape(-1, self.context_length, self.input_dim)

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
                batch_x, batch_y = (
                    batch_x.reshape(-1, self.input_dim),
                    batch_y.reshape(-1, self.input_dim),
                )
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
                    batch_x, batch_y = (
                        batch_x.reshape(-1, self.input_dim),
                        batch_y.reshape(-1, self.input_dim),
                    )
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


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_length: int,
        forecast_horizon: int,
        num_coupling: int = 1,
        hidden_dim: int = 256,
        device: str = "cpu",
        use_revin: bool = False,
    ):
        """
        Initialize Normalizing Flow based on RealNVP architecture.

        Args:
            input_dim: Input dimension
            n_components: Desired output dimension (latent space)
            num_coupling: Number of coupling layers
            hidden_dim: Width of hidden layers
            device: Device to run on
        """
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.num_coupling = num_coupling
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon

        # Create scale and translation networks for each coupling layer
        self.s_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        input_dim // 2 if i % 2 == 0 else input_dim - input_dim // 2,
                        hidden_dim,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(
                        hidden_dim,
                        input_dim - input_dim // 2 if i % 2 == 0 else input_dim // 2,
                    ),
                    nn.Tanh(),
                )
                for i in range(num_coupling)
            ]
        )

        self.t_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        input_dim // 2 if i % 2 == 0 else input_dim - input_dim // 2,
                        hidden_dim,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(
                        hidden_dim,
                        input_dim - input_dim // 2 if i % 2 == 0 else input_dim // 2,
                    ),
                )
                for i in range(num_coupling)
            ]
        )

        # Learnable scaling factors for outputs of scale networks
        self.s_scale = nn.Parameter(torch.randn(num_coupling))

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def forward(self, x, inverse: bool = False):
        """Forward pass - transform input to latent space"""
        log_det = torch.zeros(x.shape[0]).to(self.device)

        if not inverse:
            # Forward transform
            if self.use_revin:
                x = self.revin(
                    x.reshape(-1, self.context_length, self.input_dim), mode="norm"
                ).reshape(-1, self.input_dim)
            for i in range(self.num_coupling):
                x, ld = self._coupling_forward(x, i)
                log_det += ld
            return x, log_det
        else:
            # Inverse transform for sampling
            for i in reversed(range(self.num_coupling)):
                x = self._coupling_inverse(x, i)
            if self.use_revin:
                x = self.revin(
                    x.reshape(-1, self.forecast_horizon, self.input_dim), mode="denorm"
                ).reshape(-1, self.input_dim)
            return x

    def _coupling_forward(self, x, i):
        """Single coupling layer forward transform"""
        # Split input
        d = self.input_dim // 2
        x1, x2 = x[:, :d], x[:, d:]

        if i % 2 == 0:
            s = self.s_scale[i] * self.s_nets[i](x1)
            t = self.t_nets[i](x1)
            x2 = x2 * torch.exp(s) + t
        else:
            try:
                s = self.s_scale[i] * self.s_nets[i](x2)
            except RuntimeError:
                breakpoint()
            t = self.t_nets[i](x2)
            x1 = x1 * torch.exp(s) + t

        x = torch.cat([x1, x2], dim=1)
        log_det = torch.sum(s, dim=1)
        return x, log_det

    def _coupling_inverse(self, x, i):
        """Single coupling layer inverse transform"""
        d = self.input_dim // 2
        x1, x2 = x[:, :d], x[:, d:]

        if i % 2 == 0:
            s = self.s_scale[i] * self.s_nets[i](x1)
            t = self.t_nets[i](x1)
            x2 = (x2 - t) * torch.exp(-s)
        else:
            s = self.s_scale[i] * self.s_nets[i](x2)
            t = self.t_nets[i](x2)
            x1 = (x1 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=1)

    def transform(self, X):
        """Project data to latent space (numpy interface)"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z, _ = self.forward(X_tensor, inverse=False)
            return z.cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space (torch interface)"""
        z, _ = self.forward(X, inverse=False)
        return z

    def inverse_transform(self, Z):
        """Reconstruct from latent space (numpy interface)"""
        self.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            x = self.forward(Z_tensor, inverse=True)
            return x.cpu().detach().numpy()

    def inverse_transform_torch(self, Z):
        """Reconstruct from latent space (torch interface)"""
        x = self.forward(Z, inverse=True)
        return x

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        return torch.tensor(0.0, device=self.device)


class AENormalizingFlow(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        context_length: int,
        forecast_horizon: int,
        num_coupling: int = 1,
        hidden_dim: int = 256,
        device: str = "cpu",
        use_revin: bool = False,
    ):
        """
        Initialize Normalizing Flow based on RealNVP architecture.

        Args:
            input_dim: Input dimension
            n_components: Desired output dimension (latent space)
            num_coupling: Number of coupling layers
            hidden_dim: Width of hidden layers
            device: Device to run on
        """
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.num_coupling = num_coupling
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.n_components = n_components

        # Linear encoder to map into a low-dimensional space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_components),
        )
        # Linear decoder to map back to the original space
        self.decoder = nn.Sequential(
            nn.Linear(n_components, input_dim),
        )

        # Create scale and translation networks for each coupling layer
        self.s_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        n_components // 2
                        if i % 2 == 0
                        else n_components - n_components // 2,
                        hidden_dim,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(
                        hidden_dim,
                        n_components - n_components // 2
                        if i % 2 == 0
                        else n_components // 2,
                    ),
                    nn.Tanh(),
                )
                for i in range(num_coupling)
            ]
        )

        self.t_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        n_components // 2
                        if i % 2 == 0
                        else n_components - n_components // 2,
                        hidden_dim,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(
                        hidden_dim,
                        n_components - n_components // 2
                        if i % 2 == 0
                        else n_components // 2,
                    ),
                )
                for i in range(num_coupling)
            ]
        )

        # Learnable scaling factors for outputs of scale networks
        self.s_scale = nn.Parameter(torch.randn(num_coupling))

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=input_dim)

    def forward(self, x, inverse: bool = False):
        """Forward pass - transform input to latent space"""
        log_det = torch.zeros(x.shape[0]).to(self.device)

        if not inverse:
            # Forward transform
            if self.use_revin:
                x = self.revin(
                    x.reshape(-1, self.context_length, self.input_dim), mode="norm"
                ).reshape(-1, self.input_dim)
                x = self.encoder(x)
            for i in range(self.num_coupling):
                x, ld = self._coupling_forward(x, i)
                log_det += ld
            return x, log_det
        else:
            # Inverse transform for sampling
            for i in reversed(range(self.num_coupling)):
                x = self._coupling_inverse(x, i)
            if self.use_revin:
                x = self.decoder(x)
                x = self.revin(
                    x.reshape(-1, self.forecast_horizon, self.input_dim), mode="denorm"
                ).reshape(-1, self.input_dim)
            return x

    def _coupling_forward(self, x, i):
        """Single coupling layer forward transform"""
        # Split input
        d = self.n_components // 2
        x1, x2 = x[:, :d], x[:, d:]

        if i % 2 == 0:
            s = self.s_scale[i] * self.s_nets[i](x1)
            t = self.t_nets[i](x1)
            x2 = x2 * torch.exp(s) + t
        else:
            s = self.s_scale[i] * self.s_nets[i](x2)
            t = self.t_nets[i](x2)
            x1 = x1 * torch.exp(s) + t

        x = torch.cat([x1, x2], dim=1)
        log_det = torch.sum(s, dim=1)
        return x, log_det

    def _coupling_inverse(self, x, i):
        """Single coupling layer inverse transform"""
        d = self.n_components // 2
        x1, x2 = x[:, :d], x[:, d:]

        if i % 2 == 0:
            s = self.s_scale[i] * self.s_nets[i](x1)
            t = self.t_nets[i](x1)
            x2 = (x2 - t) * torch.exp(-s)
        else:
            s = self.s_scale[i] * self.s_nets[i](x2)
            t = self.t_nets[i](x2)
            x1 = (x1 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=1)

    def transform(self, X):
        """Project data to latent space (numpy interface)"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z, _ = self.forward(X_tensor, inverse=False)
            return z.cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space (torch interface)"""
        z, _ = self.forward(X, inverse=False)
        return z

    def inverse_transform(self, Z):
        """Reconstruct from latent space (numpy interface)"""
        self.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            x = self.forward(Z_tensor, inverse=True)
            return x.cpu().detach().numpy()

    def inverse_transform_torch(self, Z):
        """Reconstruct from latent space (torch interface)"""
        x = self.forward(Z, inverse=True)
        return x

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        return torch.tensor(0.0, device=self.device)


class JustRevIn(nn.Module):
    def __init__(
        self,
        num_features: int,
        context_length: int,
        forecast_horizon: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_features = num_features
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon

        self.revin = RevIN(num_features=num_features)

    def forward(self, x, mode: str):
        if mode == "norm":
            return self.revin(
                x.reshape(-1, self.context_length, self.num_features), mode="norm"
            ).reshape(-1, self.num_features)
        elif mode == "denorm":
            return self.revin(
                x.reshape(-1, self.forecast_horizon, self.num_features), mode="denorm"
            ).reshape(-1, self.num_features)
        else:
            raise ValueError("Invalid mode. Must be 'norm' or 'denorm'")

    def transform(self, X):
        """Project data to latent space (numpy interface)"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            z = self.forward(X_tensor, mode="norm")
            return z.cpu().detach().numpy()

    def transform_torch(self, X):
        """Project data to latent space (torch interface)"""
        z = self.forward(X, mode="norm")
        return z

    def inverse_transform(self, Z):
        """Reconstruct from latent space (numpy interface)"""
        self.eval()
        with torch.no_grad():
            Z_tensor = torch.FloatTensor(Z).to(self.device)
            x = self.forward(Z_tensor, mode="denorm")
            return x.cpu().detach().numpy()

    def inverse_transform_torch(self, Z):
        """Reconstruct from latent space (torch interface)"""
        x = self.forward(Z, mode="denorm")
        return x

    def reconstruction_loss(self, X_batch):
        """Compute reconstruction loss"""
        return torch.tensor(0.0, device=self.device)
