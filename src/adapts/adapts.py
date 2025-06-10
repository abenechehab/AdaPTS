from typing import TYPE_CHECKING, List, Optional, Tuple
import copy
import os
from tqdm import tqdm
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

from adapts.utils.calibration import compute_ks_metric
from adapts.utils.preprocessing import AxisScaler, AxisPCA, get_gpu_memory_stats
from adapts.adapters import IdentityTransformer

if TYPE_CHECKING:
    from adapts.icl.iclearner import ICLTrainer, ICLObject
    from adapts.adapters import MultichannelProjector
from torch.utils.tensorboard import SummaryWriter


class ADAPTS:
    def __init__(
        self,
        adapter: "MultichannelProjector",
        iclearner: "ICLTrainer",
        n_features: int,
        n_components: int,
        pca_in_preprocessing: bool = False,
        scaler_in_preprocessing: bool = True,
    ):
        """
        Initialize the ADAPTS model with the specified adapter, model, and
        hyperparameters.

        Args:
            adapter (MultichannelProjector): adapter model.
            iclearner (ICLTrainer): ICLearner model.
            n_features (int): Number of features in the input time series.
            n_components (int): Number of components in the latent space.
            pca_in_preprocessing (bool, optional): Whether to use PCA in the
                preprocessing pipeline. Defaults to False.
            scaler_in_preprocessing (bool, optional): Whether to use a scaler in the
                preprocessing pipeline. Defaults to True.
        """

        self.n_features = n_features
        self.n_components = n_components

        if pca_in_preprocessing and scaler_in_preprocessing:
            self.scaler = make_pipeline(
                AxisScaler(MinMaxScaler(), axis=1),
                AxisScaler(StandardScaler(), axis=1),
                AxisPCA(n_components=n_features, axis=1),
            )
        elif scaler_in_preprocessing:
            self.scaler = make_pipeline(
                AxisScaler(MinMaxScaler(), axis=1),
                AxisScaler(StandardScaler(), axis=1),
            )
        else:
            self.scaler = IdentityTransformer()

        self.adapter = adapter

        self.iclearner = iclearner

    def fit_adapter(self, X: NDArray):
        """
        Fit the adapter on the input data.

        Args:
            X (NDArray): Input time series data.
        """
        self.scaler.fit(X)
        self.adapter.fit(self.scaler.transform(X))

    def fine_tune_iclearner(
        self,
        X: NDArray[np.float32],  # input sequences
        y: NDArray[np.float32],  # target sequences
        X_val: Optional[NDArray[np.float32]] = None,  # input sequences
        y_val: Optional[NDArray[np.float32]] = None,  # target sequences
        n_epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 5.0,
        verbose: int = 0,
        seed: int = 13,
        use_adapter: bool = True,
        logger=None,
    ):
        """Fine-tune the time series foundation model on the given time series data

        Args:
            X (NDArray[np.float32]): Input sequences
            y (NDArray[np.float32]): Target sequences
            X_val (Optional[NDArray[np.float32]], optional): Validation input sequences
            y_val (Optional[NDArray[np.float32]], optional): Validation target sequences
            n_epochs (int, optional): Number of epochs. Defaults to 1.
            batch_size (int, optional): Batch size. Defaults to 8.
            learning_rate (float, optional): Learning rate. Defaults to 1e-4.
            max_grad_norm (float, optional): Maximum gradient norm. Defaults to 5.0.
            verbose (int, optional): Verbosity level. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 13.
            use_adapter (bool, optional): Whether to use the adapter. Defaults to True.
            logger ([type], optional): Logger. Defaults to None.
        """

        if use_adapter:
            direct_transform = self.adapter.transform_torch
            inverse_transform = self.adapter.inverse_transform_torch
            self.adapter.base_projector_.eval()
        else:
            direct_transform = torch.nn.Identity()
            inverse_transform = torch.nn.Identity()

        X = self.scaler.fit_transform(X)
        y = self.scaler.transform(y)

        self.iclearner.fine_tune(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            seed=seed,
            inverse_transform=inverse_transform,
            direct_transform=direct_transform,
            logger=logger,
        )

    def transform(self, X: NDArray) -> NDArray:
        """
        Transform the input data using the adapter.

        Args:
            X (NDArray): Input time series data to be transformed.

        Returns:
            NDArray: Transformed data.
        """
        return self.adapter.transform(self.scaler.fit_transform(X))

    def inverse_transform(self, X_transformed: NDArray) -> NDArray:
        """
        Inverse transform the data back to its original representation.

        Args:
            X_transformed (NDArray): Transformed time series data.

        Returns:
            NDArray: Data transformed back to the original space.
        """

        if hasattr(self.adapter.base_projector_, "likelihood"):
            return self.scaler.inverse_transform(
                self.adapter.inverse_transform(X_transformed)[0]
            )
        return self.scaler.inverse_transform(
            self.adapter.inverse_transform(X_transformed)
        )

    def predict_multi_step(
        self, X: NDArray, prediction_horizon: int, n_samples: int = 10, **kwargs
    ) -> Tuple[NDArray, ...]:
        """
        Perform multi-step prediction for a given horizon.

        Args:
            X (NDArray): Input time series data.
            prediction_horizon (int): Prediction horizon.
            n_samples (int, optional): Number of samples to draw. Defaults to 10.

        Returns:
            Tuple[NDArray, ...]: Mean, mode, lower bound, and upper bound of the
                predictions.
        """
        assert X.shape[1] == self.n_features, (
            f"N features doesnt correspond to {self.n_features}"
        )

        self.context_length = X.shape[-1]
        self.X = X
        self.prediction_horizon = prediction_horizon

        # train mode in order for dropout to be stochastic
        if hasattr(self.adapter.base_projector_, "train"):
            self.adapter.base_projector_.train()

        all_predictions = []
        for _ in range(n_samples):
            # Step 1: Transform the time series (stochasticity is here)
            X_transformed = self.transform(X[:, :, :-prediction_horizon])

            # Step 2: Perform time series forecasting multiple times
            self.iclearner.update_context(
                time_series=copy.copy(X_transformed),
                context_length=X_transformed.shape[-1],
            )

            self.icl_object: List["ICLObject"] = self.iclearner.predict_long_horizon(
                prediction_horizon=prediction_horizon,
                **kwargs,
            )

            predictions = []
            for dim in range(self.n_components):
                predictions.append(self.icl_object[dim].mean_arr)
            all_predictions.append(predictions)

        # Convert to numpy array for statistics
        all_predictions = np.array(
            all_predictions
        )  # [n_samples, n_components, batch, horizon]

        self.all_predictions = all_predictions

        # Update ICL objects with sample statistics
        for dim in range(self.n_components):
            dim_predictions = all_predictions[:, dim]
            self.icl_object[dim].mean_arr = np.mean(dim_predictions, axis=0)
            self.icl_object[dim].mode_arr = np.mean(dim_predictions, axis=0)
            self.icl_object[dim].sigma_arr = np.std(dim_predictions, axis=0)

        # Step 3: Inverse transform the predictions
        all_mean = []
        all_mode = []
        all_lb = []
        all_ub = []
        for dim in range(self.n_components):
            # -------------------- Useful for Plots --------------------
            mode_arr = self.icl_object[dim].mode_arr
            mean_arr = self.icl_object[dim].mean_arr
            sigma_arr = self.icl_object[dim].sigma_arr

            all_mean.append(mean_arr)
            all_mode.append(mode_arr)
            all_lb.append(mean_arr - sigma_arr)
            all_ub.append(mean_arr + sigma_arr)

        self.mean = self.inverse_transform(np.concatenate(all_mean, axis=1))
        self.mode = self.inverse_transform(np.concatenate(all_mode, axis=1))
        self.lb = self.inverse_transform(np.concatenate(all_lb, axis=1))
        self.ub = self.inverse_transform(np.concatenate(all_ub, axis=1))

        return self.mean, self.mode, self.lb, self.ub

    def compute_metrics(self, calibration: bool = True, logdir: Optional[Path] = None):
        """
        Compute the prediction metrics such as MSE and KS test.

        Args:
            burnin (int, optional): Number of initial steps to ignore when computing
                metrics. Defaults to 0.

        Returns:
            dict: Dictionary containing various prediction metrics.
        """
        metrics = {}

        # ------- MSE --------
        metrics["mse"] = torch.nn.MSELoss()(
            torch.tensor(self.X[:, :, -self.prediction_horizon :]),
            torch.tensor(self.mean),
        ).item()
        # ------- MAE --------
        metrics["mae"] = torch.nn.L1Loss()(
            torch.tensor(self.X[:, :, -self.prediction_horizon :]),
            torch.tensor(self.mean),
        ).item()

        # ------- scaled MSE --------
        scaled_groundtruth = self.scaler.transform(
            self.X[:, :, -self.prediction_horizon :]
        )
        scaled_mean = self.scaler.transform(self.mean)
        metrics["scaled_mse"] = torch.nn.MSELoss()(
            torch.tensor(scaled_groundtruth), torch.tensor(scaled_mean)
        ).item()
        metrics["scaled_mae"] = torch.nn.L1Loss()(
            torch.tensor(scaled_groundtruth), torch.tensor(scaled_mean)
        ).item()

        if calibration:
            # ------ KS -------
            kss, ece, ks_quantiles = compute_ks_metric(
                groundtruth=self.X[:, :, -self.prediction_horizon :],
                all_predictions=self.all_predictions,
                n_features=self.n_features,
                inverse_transform=self.inverse_transform,
            )

            # metrics["perdim_ks"] = kss
            metrics["ks"] = kss.mean()
            metrics["ece"] = ece.mean()

            if logdir:
                np.save(logdir / "ks.npy", kss)
                np.save(logdir / "ece.npy", ece)
                np.save(logdir / "ks_quantiles.npy", ks_quantiles)

        return metrics

    def plot_multi_step(
        self,
        feature_names: Optional[List[str]] = None,
        xlim: Optional[List[float]] = None,
        savefigpath: Optional[str] = None,
        sample: int = 0,
    ):
        """
        Plot multi-step predictions and ground truth.

        Args:
            feature_names (Optional[List[str]], optional): Names of the features.
                Defaults to None.
            xlim (Optional[List[float]], optional): X-axis limits.
                Defaults to None.
            savefigpath (Optional[str], optional): File path to save the plot.
                Defaults to None.
        """
        if not feature_names:
            feature_names = [f"f{i}" for i in range(self.n_features)]

        _, axes = plt.subplots(
            (self.n_features // 3) + 1,
            3,
            figsize=(20, 25),
            gridspec_kw={"hspace": 0.3},
            sharex=True,
        )
        axes = list(np.array(axes).flatten())
        for dim in range(self.n_features):
            ax = axes[dim]
            ax.plot(
                np.arange(self.context_length),
                self.X[sample, dim, :],
                color="blue",
                linewidth=1,
                label="groundtruth",
                # linestyle="--",
            )
            ax.plot(
                np.arange(
                    self.context_length - self.prediction_horizon - 1,
                    self.context_length - 1,
                ),
                self.mean[sample, dim, -self.prediction_horizon :],
                label="multi-step",
                color=sns.color_palette("colorblind")[1],
            )
            ax.set_ylabel(feature_names[dim], rotation=0, labelpad=20)
            ax.set_yticklabels([])
            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([0, self.context_length - 1])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=6)
        if savefigpath:
            plt.savefig(savefigpath, bbox_inches="tight")
        plt.show()

    def adapter_supervised_fine_tuning(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        coeff_reconstruction=0.0,
        n_epochs=300,
        learning_rate=0.001,
        batch_size=32,
        max_patience=10,
        device="cpu",
        log_dir="logs/",
        reverse=False,
        verbose=0,
        logger=None,
    ):
        """
        Trains the adapter using supervised learning.
        This method trains the adapter using supervised learning by minimizing the
        prediction error of an in-context learner on a given dataset. The training
        process includes early stopping and learning rate scheduling based on
        validation performance.

        Args:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target data.
            X_val (np.ndarray, optional): Validation input data. If None, 20% of
                training data will be used.
            y_val (np.ndarray, optional): Validation target data. If None, 20% of
                training data will be used.
            coeff_reconstruction (float, optional): Coefficient for reconstruction
                loss. Defaults to 0.0.
            n_epochs (int, optional): Maximum number of training epochs.
                Defaults to 300.
            learning_rate (float, optional): Initial learning rate.
                Defaults to 0.001.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            max_patience (int, optional): Number of epochs to wait for improvement
                before early stopping. Defaults to 10.
            device (str, optional): Device to use for training ('cpu' or 'cuda').
                Defaults to "cpu".
            log_dir (str, optional): Directory for tensorboard logs.
                Defaults to "logs/".
            reverse (bool, optional): If True, maximizes loss instead of minimizing.
                    Defaults to False.
            verbose (int, optional): Verbosity level. Defaults to 0.
            logger (logging.Logger, optional): Logger instance for logging messages.
                Defaults to None.
        Raises:
            ValueError: If the specified log directory does not exist.
        Returns:
            None: The method updates the adapter in-place.
        Note:
            - The method uses TensorBoard for logging training metrics
            - Implements early stopping based on validation loss
            - Uses learning rate scheduling with ReduceLROnPlateau
            - Supports both deterministic and probabilistic predictions
            - Handles both reconstruction and KL divergence losses if applicable
        """
        if not os.path.exists(log_dir):
            raise ValueError(f"Log directory {log_dir} does not exist")

        writer = SummaryWriter(log_dir)

        assert isinstance(self.adapter.base_projector_, torch.nn.Module), (
            "adapter must be a PyTorch Module"
        )

        self.scaler.fit(np.concatenate([X_train, y_train], axis=-1))
        X_scaled, y_scaled = (
            self.scaler.transform(X_train),
            self.scaler.transform(y_train),
        )

        # Create dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32),
        )

        # Split into train and validation sets (80-20 split)
        if (X_val is None) or (y_val is None):
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
        else:
            X_val_scaled, y_val_scaled = (
                self.scaler.transform(X_val),
                self.scaler.transform(y_val),
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_scaled, dtype=torch.float32),
            )
            val_size = len(val_dataset)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        self.iclearner.eval()

        def make_predictions(X_batch, y_batch):
            X_batch_transformed = self.adapter.transform_torch(X_batch)

            self.iclearner.update_context(
                time_series=X_batch_transformed,
                context_length=X_batch_transformed.shape[-1],
            )

            icl_predictions = self.iclearner.predict_long_horizon(
                prediction_horizon=y_batch.shape[-1],
                batch_size=batch_size,
                verbose=0,
            )

            all_means = []
            for dim in range(self.n_components):
                all_means.append(icl_predictions[dim].predictions)

            predictions = self.adapter.inverse_transform_torch(
                torch.concat(all_means, axis=1)
            )

            return predictions

        self.adapter.base_projector_.train()

        optimizer = torch.optim.Adam(
            self.adapter.base_projector_.parameters(), lr=learning_rate
        )
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.75, patience=5, min_lr=1e-6
        )

        for epoch in tqdm(
            range(n_epochs),
            desc="Training Epochs",
            disable=not bool(verbose),
        ):
            # log gpu memory
            gpu_stats = get_gpu_memory_stats()
            for key, value in gpu_stats.items():
                writer.add_scalar(f"gpu/{key}", value, epoch)

            total_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()

                X_batch, y_batch = (
                    X_batch.to(torch.device(device)),
                    y_batch.to(torch.device(device)),
                )

                predictions = make_predictions(X_batch=X_batch, y_batch=y_batch)

                if hasattr(self.adapter.base_projector_, "likelihood"):
                    mu, logvar = predictions
                    predicted_dist = torch.distributions.Normal(
                        mu, torch.exp(0.5 * logvar)
                    )
                    pred_loss = -predicted_dist.log_prob(y_batch).mean()
                else:
                    criterion = torch.nn.MSELoss()
                    pred_loss = criterion(predictions, y_batch)

                loss = pred_loss
                if coeff_reconstruction > 0:
                    # reconstruction loss
                    reconstruction_loss = (
                        self.adapter.base_projector_.reconstruction_loss(
                            X_batch.permute(0, 2, 1).reshape(-1, X_batch.shape[1])
                        )
                    )
                    loss += coeff_reconstruction * reconstruction_loss
                else:
                    if hasattr(self.adapter.base_projector_, "kl_loss"):
                        kl_loss = self.adapter.base_projector_.kl_loss(X_batch=X_batch)
                        loss += kl_loss
                        writer.add_scalar(
                            "Loss/KL",
                            kl_loss.item(),
                            epoch * len(train_loader) + batch_idx,
                        )

                if reverse:
                    loss = -loss

                loss.backward()
                # Log gradient norm of model parameters
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.adapter.base_projector_.parameters(), float("inf")
                )
                writer.add_scalar(
                    "Gradients/norm",
                    grad_norm.item(),
                    epoch * len(train_loader) + batch_idx,
                )
                optimizer.step()

                total_loss += loss.item()

                # Log batch loss
                writer.add_scalar(
                    "Loss/batch", loss.item(), epoch * len(train_loader) + batch_idx
                )
                if coeff_reconstruction > 0:
                    writer.add_scalar(
                        "Loss/batch_recon",
                        reconstruction_loss.item(),
                        epoch * len(train_loader) + batch_idx,
                    )
                writer.add_scalar(
                    "Loss/batch_pred",
                    pred_loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )

            avg_loss = total_loss * batch_size / len(train_dataset)
            # Log epoch metrics
            writer.add_scalar("Loss/training", avg_loss, epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

            # Compute validation loss
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = (
                        X_val.to(torch.device(device)),
                        y_val.to(torch.device(device)),
                    )
                    val_predictions = make_predictions(X_batch=X_val, y_batch=y_val)
                    if hasattr(self.adapter.base_projector_, "likelihood"):
                        mu, logvar = val_predictions
                        predicted_dist = torch.distributions.Normal(
                            mu, torch.exp(0.5 * logvar)
                        )
                        val_loss += -predicted_dist.log_prob(y_val).mean().item()
                    else:
                        val_loss += criterion(val_predictions, y_val).item()
            val_loss = val_loss * batch_size / val_size
            if reverse:
                val_loss = -val_loss

            # Log validation loss
            writer.add_scalar("Loss/validation", val_loss, epoch)

            # Use scheduler for learning rate adjustment based on validation loss
            scheduler.step(val_loss)

            # Early stopping based on validation loss
            if epoch == 0:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(
                    self.adapter.base_projector_.state_dict()
                )
                best_epoch = epoch
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(
                        self.adapter.base_projector_.state_dict()
                    )
                    best_epoch = epoch
                else:
                    patience_counter += 1

            if patience_counter >= max_patience:  # Early stopping patience
                if logger:
                    logger.info(f"Early stopping at epoch {epoch}")
                else:
                    print(f"Early stopping at epoch {epoch}")
                break
        if logger:
            logger.info(f"Restoring weights from epoch {best_epoch}")
        else:
            print(f"Restoring weights from epoch {best_epoch}")
        self.adapter.base_projector_.load_state_dict(best_model_weights)
        del best_model_weights
        self.adapter.base_projector_.eval()
        writer.close()
        return

    def adapter_and_head_supervised_fine_tuning(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        coeff_reconstruction=0.0,
        n_epochs=300,
        learning_rate=0.001,
        batch_size=32,
        early_stopping_patience=10,
        device="cpu",
        log_dir="logs/",
        reverse=False,
        verbose=0,
    ):
        """
        Jointly Trains the adapter and any learnable part of the Foundation Model using
        supervised learning.
        This method trains the adapter using supervised learning by minimizing the
        prediction error of an in-context learner on a given dataset. The training
        process includes early stopping and learning rate scheduling based on
        validation performance.

        Args:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target data.
            X_val (np.ndarray, optional): Validation input data. If None, 20% of
                training data will be used.
            y_val (np.ndarray, optional): Validation target data. If None, 20% of
                training data will be used.
            coeff_reconstruction (float, optional): Coefficient for reconstruction
                loss. Defaults to 0.0.
            n_epochs (int, optional): Maximum number of training epochs.
                Defaults to 300.
            learning_rate (float, optional): Initial learning rate.
                Defaults to 0.001.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            max_patience (int, optional): Number of epochs to wait for improvement
                before early stopping. Defaults to 10.
            device (str, optional): Device to use for training ('cpu' or 'cuda').
                Defaults to "cpu".
            log_dir (str, optional): Directory for tensorboard logs.
                Defaults to "logs/".
            reverse (bool, optional): If True, maximizes loss instead of minimizing.
                    Defaults to False.
            verbose (int, optional): Verbosity level. Defaults to 0.
        Raises:
            ValueError: If the specified log directory does not exist.
        Returns:
            None: The method updates the adapter in-place.
        Note:
            - The method uses TensorBoard for logging training metrics
            - Implements early stopping based on validation loss
            - Uses learning rate scheduling with ReduceLROnPlateau
            - Supports both deterministic and probabilistic predictions
            - Handles both reconstruction and KL divergence losses if applicable
        """
        if not os.path.exists(log_dir):
            raise ValueError(f"Log directory {log_dir} does not exist")

        writer = SummaryWriter(log_dir)

        assert isinstance(self.adapter.base_projector_, torch.nn.Module), (
            "adapter must be a PyTorch Module"
        )

        self.scaler.fit(np.concatenate([X_train, y_train], axis=-1))
        X_scaled, y_scaled = (
            self.scaler.transform(X_train),
            self.scaler.transform(y_train),
        )

        # Create dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32),
        )

        # Split into train and validation sets (80-20 split)
        if (X_val is None) or (y_val is None):
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
        else:
            X_val_scaled, y_val_scaled = (
                self.scaler.transform(X_val),
                self.scaler.transform(y_val),
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_scaled, dtype=torch.float32),
            )
            val_size = len(val_dataset)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        def make_predictions(X_batch, y_batch):
            X_batch_transformed = self.adapter.transform_torch(X_batch)

            self.iclearner.update_context(
                time_series=X_batch_transformed,
                context_length=X_batch_transformed.shape[-1],
            )

            icl_predictions = self.iclearner.predict_long_horizon(
                prediction_horizon=y_batch.shape[-1],
                batch_size=batch_size,
                verbose=0,
            )

            all_means = []
            for dim in range(self.n_components):
                all_means.append(icl_predictions[dim].predictions)

            predictions = self.adapter.inverse_transform_torch(
                torch.concat(all_means, axis=1)
            )

            return predictions

        self.adapter.base_projector_.train()
        self.iclearner.model.train()

        optimizer = torch.optim.Adam(
            list(self.adapter.base_projector_.parameters())
            + list(self.iclearner.model.parameters()),
            lr=learning_rate,
        )
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        # Track best validation loss for early stopping
        best_val_loss = float("inf")

        for epoch in tqdm(
            range(n_epochs),
            desc="Training Epochs",
            disable=not bool(verbose),
        ):
            # log gpu memory
            gpu_stats = get_gpu_memory_stats()
            for key, value in gpu_stats.items():
                writer.add_scalar(f"gpu/{key}", value, epoch)

            total_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                self.adapter.base_projector_.train()
                self.iclearner.model.train()
                optimizer.zero_grad()

                X_batch, y_batch = (
                    X_batch.to(torch.device(device)),
                    y_batch.to(torch.device(device)),
                )

                predictions = make_predictions(X_batch=X_batch, y_batch=y_batch)

                criterion = torch.nn.MSELoss()
                pred_loss = criterion(predictions, y_batch)

                loss = pred_loss
                if coeff_reconstruction > 0:
                    # reconstruction loss
                    reconstruction_loss = (
                        self.adapter.base_projector_.reconstruction_loss(
                            X_batch.permute(0, 2, 1).reshape(-1, X_batch.shape[1])
                        )
                    )
                    loss += coeff_reconstruction * reconstruction_loss

                if reverse:
                    loss = -loss

                loss.backward()
                # Log gradient norm of model parameters
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.adapter.base_projector_.parameters(), float("inf")
                )
                writer.add_scalar(
                    "Gradients/norm(adapters)",
                    grad_norm.item(),
                    epoch * len(train_loader) + batch_idx,
                )
                model_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.iclearner.model.head.parameters(), float("inf")
                )
                writer.add_scalar(
                    "Gradients/norm(model)",
                    model_grad_norm.item(),
                    epoch * len(train_loader) + batch_idx,
                )
                optimizer.step()

                total_loss += loss.item()

                # Log batch loss
                writer.add_scalar(
                    "Loss/batch", loss.item(), epoch * len(train_loader) + batch_idx
                )
                if coeff_reconstruction > 0:
                    writer.add_scalar(
                        "Loss/batch_recon",
                        reconstruction_loss.item(),
                        epoch * len(train_loader) + batch_idx,
                    )
                writer.add_scalar(
                    "Loss/batch_pred",
                    pred_loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )

            avg_loss = total_loss * batch_size / len(train_dataset)
            # Log epoch metrics
            writer.add_scalar("Loss/training", avg_loss, epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

            # Compute validation loss
            self.adapter.base_projector_.eval()
            self.iclearner.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = (
                        X_val.to(torch.device(device)),
                        y_val.to(torch.device(device)),
                    )
                    val_predictions = make_predictions(X_batch=X_val, y_batch=y_val)
                    val_loss += criterion(val_predictions, y_val).item()
            val_loss = val_loss * batch_size / val_size
            if reverse:
                val_loss = -val_loss

            # Log validation loss
            writer.add_scalar("Loss/validation", val_loss, epoch)

            # Use scheduler for learning rate adjustment based on validation loss
            scheduler.step(val_loss)

            # Early stopping check
            patience_counter = 0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        self.adapter.base_projector_.eval()
        self.iclearner.model.eval()
        writer.close()
        return
