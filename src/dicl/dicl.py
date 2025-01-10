from typing import TYPE_CHECKING, List, Optional, Tuple
import copy

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

from dicl.utils.preprocessing import AxisScaler, get_gpu_memory_stats

if TYPE_CHECKING:
    from dicl.icl.iclearner import ICLTrainer, ICLObject
    from dicl.adapters import MultichannelProjector
from torch.utils.tensorboard import SummaryWriter


class DICL:
    def __init__(
        self,
        disentangler: "MultichannelProjector",
        iclearner: "ICLTrainer",
        n_features: int,
        n_components: int,
    ):
        """
        Initialize the DICL model with the specified disentangler, model, and
        hyperparameters.

        Args:
            disentangler (Any): The disentangler or preprocessor pipeline.
            iclearner (ICLTrainer): The ICLTrainer instance used for time series
            n_features (int): Number of input features.
            n_components (int): Number of independent components to be learned.
            model (AutoModel): Pretrained model used for time series prediction.
            tokenizer (AutoTokenizer): Tokenizer associated with the pretrained model.
            rescale_factor (float, optional): Rescaling factor for the transformed data.
                Defaults to 7.0.
            up_shift (float, optional): Shift factor applied to rescaled data.
                Defaults to 1.5.
        """

        self.n_features = n_features
        self.n_components = n_components

        self.scaler = make_pipeline(
            AxisScaler(MinMaxScaler(), axis=1),
            AxisScaler(StandardScaler(), axis=1),
        )

        # self.disentangler = make_pipeline(self.scaler, disentangler)
        self.disentangler = disentangler

        self.iclearner = iclearner

    def fit_disentangler(self, X: NDArray):
        """
        Fit the disentangler on the input data.

        Args:
            X (NDArray): Input time series data.
        """
        self.scaler.fit(X)
        self.disentangler.fit(self.scaler.transform(X))

    def fine_tune_iclearner(
        self,
        X: NDArray[np.float32],  # input sequences
        y: NDArray[np.float32],  # target sequences
        n_epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 5.0,
        verbose: int = 0,
        seed: int = 13,
    ):
        """Fine-tune the model on the given time series data

        Args:
            X: Input sequences of shape [n_samples, n_features, input_length]
            y: Target sequences of shape [n_samples, n_features, forecast_horizon]
            n_epochs: Number of epochs to train
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            verbose: verbosity level
        """

        X = self.transform(X)
        y = self.transform(y)

        self.iclearner.fine_tune(
            X=X,
            y=y,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            seed=seed,
        )

    def transform(self, X: NDArray) -> NDArray:
        """
        Transform the input data using the disentangler.

        Args:
            X (NDArray): Input time series data to be transformed.

        Returns:
            NDArray: Transformed data.
        """
        return self.disentangler.transform(self.scaler.transform(X))

    def inverse_transform(self, X_transformed: NDArray) -> NDArray:
        """
        Inverse transform the data back to its original representation.

        Args:
            X_transformed (NDArray): Transformed time series data.

        Returns:
            NDArray: Data transformed back to the original space.
        """
        return self.scaler.inverse_transform(
            self.disentangler.inverse_transform(X_transformed)
        )

    def predict_multi_step(
        self, X: NDArray, prediction_horizon: int, **kwargs
    ) -> Tuple[NDArray, ...]:
        """
        Perform multi-step prediction for a given horizon.

        Args:
            X (NDArray): Input time series data.
            prediction_horizon (int): Number of steps to predict into the future
                (taken from the end of the input time series).
            stochastic (bool, optional): Whether to apply stochastic predictions.
                Defaults to False.
            if_true_mean_else_mode (bool, optional): Whether to return mean predictions
                (True) or mode predictions (False). Defaults to False.

        Returns:
            Tuple[NDArray, ...]: Mean, mode, lower bound, and upper bound of the
                predictions.
        """
        assert (
            X.shape[1] == self.n_features
        ), f"N features doesnt correspond to {self.n_features}"

        self.context_length = X.shape[-1]
        self.X = X
        self.prediction_horizon = prediction_horizon

        # Step 1: Transform the time series
        X_transformed = self.transform(X[:, :, :-prediction_horizon])

        # Step 2: Perform time series forecasting
        self.iclearner.update_context(
            time_series=copy.copy(X_transformed),
            context_length=X_transformed.shape[-1],
        )

        self.icl_object: List["ICLObject"] = self.iclearner.predict_long_horizon(
            prediction_horizon=prediction_horizon,
            **kwargs,
        )

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

    def compute_metrics(self):
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
        batch_size=16,
        early_stopping_patience=10,
        device="cpu",
        log_dir="logs/",
    ):
        writer = SummaryWriter(log_dir)

        assert isinstance(
            self.disentangler.base_projector_, torch.nn.Module
        ), "Disentangler must be a PyTorch Module"

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
            X_batch_transformed = self.disentangler.transform_torch(X_batch)

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

            predictions = self.disentangler.inverse_transform_torch(
                torch.concat(all_means, axis=1)
            )

            return predictions

        self.disentangler.base_projector_.train()

        optimizer = torch.optim.Adam(
            self.disentangler.base_projector_.parameters(), lr=learning_rate
        )
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        # Track best validation loss for early stopping
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
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

                criterion = torch.nn.MSELoss()
                loss = criterion(predictions, y_batch)

                # reconstruction loss
                reconstruction_loss = (
                    self.disentangler.base_projector_.reconstruction_loss(
                        X_batch.permute(0, 2, 1).reshape(-1, X_batch.shape[1])
                    )
                )
                loss += coeff_reconstruction * reconstruction_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Log batch loss
                writer.add_scalar(
                    "Loss/batch", loss.item(), epoch * len(train_loader) + batch_idx
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
                    val_loss += criterion(val_predictions, y_val).item()
            val_loss = val_loss * batch_size / val_size

            # Log validation loss
            writer.add_scalar("Loss/validation", val_loss, epoch)

            # Use scheduler for learning rate adjustment based on validation loss
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        self.disentangler.base_projector_.eval()
        writer.close()
        return
