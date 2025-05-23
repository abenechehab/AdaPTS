from typing import TYPE_CHECKING, Optional, List

import copy
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import torch
from torch.optim.lr_scheduler import OneCycleLR

from momentfm.utils.utils import control_randomness

if TYPE_CHECKING:
    from momentfm import MOMENTPipeline

from adapts.icl.iclearner import ICLTrainer, ICLObject


class MomentICLTrainer(ICLTrainer):
    def __init__(
        self, model: "MOMENTPipeline", n_features: int, forecast_horizon: int = 96
    ):
        """
        MomentICLTrainer is an implementation of ICLTrainer using the MOMENT
        foundation model for time series forecasting.

        Args:
            n_features (int): Number of features in the time series data
            forecast_horizon (int): Number of steps to forecast
            rescale_factor (float): Rescaling factor for data normalization
            up_shift (float): Shift value applied after rescaling
        """

        self.model = model

        self.n_features = n_features
        self.forecast_horizon = forecast_horizon

        self.icl_object: List[ICLObject] = [ICLObject() for _ in range(self.n_features)]

    def update_context(
        self,
        time_series: NDArray[np.float32] | torch.Tensor,
        context_length: Optional[int] = None,
    ):
        """Updates the context with given time series data"""
        if context_length is not None:
            self.context_length = context_length
        else:
            self.context_length = time_series.shape[-1]

        assert len(time_series.shape) == 3 and time_series.shape[1] == self.n_features

        # Store original time series for each feature
        for dim in range(self.n_features):
            self.icl_object[dim].time_series = time_series[
                :, dim, : self.context_length
            ]

        return self.icl_object

    def compute_statistics(self):
        """Compute statistics on predictions"""
        for dim in range(self.n_features):
            # MOMENT provides point estimates, so mean=mode=prediction, sigma=0
            preds = self.icl_object[dim].predictions
            self.icl_object[dim].mean_arr = preds.cpu().detach().numpy()
            self.icl_object[dim].mode_arr = preds.cpu().detach().numpy()
            self.icl_object[dim].sigma_arr = np.zeros_like(preds.cpu().detach().numpy())
        return self.icl_object

    def predict_long_horizon(
        self,
        prediction_horizon: int,
        batch_size: int = 128,
        native_multivariate: bool = False,
        verbose: int = 1,
    ):
        """Multi-step prediction using MOMENT model"""
        # Get device from model
        self.model.eval()
        device = next(self.model.parameters()).device
        if native_multivariate:
            # Process all features together
            tensor_ts = torch.cat(
                [
                    self.icl_object[dim].time_series
                    if isinstance(self.icl_object[dim].time_series, torch.Tensor)
                    else torch.from_numpy(self.icl_object[dim].time_series)
                    .unsqueeze(1)
                    .float()
                    .to(device)
                    for dim in range(self.n_features)
                ],
                axis=1,
            )
            # Process in batches to avoid memory issues
            all_predictions = []
            for i in tqdm(range(0, tensor_ts.shape[0], batch_size), desc="batch"):
                batch_end = min(i + batch_size, tensor_ts.shape[0])
                batch_ts = tensor_ts[i:batch_end]
                batch_predictions = self.model(x_enc=batch_ts).forecast
                all_predictions.append(batch_predictions)

            # Stack all batches together
            predictions = torch.concatenate(all_predictions, axis=0)

            for dim in range(self.n_features):
                self.icl_object[dim].predictions = predictions[:, dim, :].unsqueeze(1)
        else:
            for dim in tqdm(
                range(self.n_features),
                desc="feature",
                disable=not bool(verbose),
            ):
                ts = self.icl_object[dim].time_series
                tensor_ts = ts if isinstance(ts, torch.Tensor) else torch.from_numpy(ts)
                tensor_ts = tensor_ts.float().to(device)
                # takes in tensor of shape [batchsize, n_channels, context_length]
                tensor_ts = tensor_ts.unsqueeze(1)
                # Process in batches to avoid memory issues
                all_predictions = []
                for i in tqdm(
                    range(0, ts.shape[0], batch_size),
                    desc=f"batch on feature {dim}:",
                    disable=not bool(verbose),
                ):
                    batch_end = min(i + batch_size, ts.shape[0])
                    batch_ts = tensor_ts[i:batch_end]
                    batch_predictions = self.model(x_enc=batch_ts).forecast
                    all_predictions.append(batch_predictions)
                # Stack all batches together
                predictions = torch.concat(all_predictions, axis=0)
                self.icl_object[dim].predictions = predictions
        return self.compute_statistics()

    def fine_tune(
        self,
        X: NDArray[np.float32],  # input sequences
        y: NDArray[np.float32],  # target sequences
        X_val: Optional[NDArray[np.float32]] = None,
        y_val: Optional[NDArray[np.float32]] = None,
        n_epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 5.0,
        verbose: int = 0,
        seed: int = 13,
        inverse_transform: callable = torch.nn.Identity(),
        direct_transform: callable = torch.nn.Identity(),
        max_patience: int = 5,
        logger=None,
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

        self.model.train()

        # Set random seeds for PyTorch, Numpy etc.
        control_randomness(seed=seed)

        # Get device from model
        device = next(self.model.parameters()).device

        # Create dataset and data loader
        tensor_X = torch.from_numpy(X).float()
        tensor_y = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)

        if (X_val is not None) and (y_val is not None):
            tensor_X_val = torch.from_numpy(X_val).float()
            tensor_y_val = torch.from_numpy(y_val).float()
            val_dataset = torch.utils.data.TensorDataset(tensor_X_val, tensor_y_val)
            train_dataset = dataset
        else:
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Setup training
        criterion = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()

        # Create a OneCycleLR scheduler
        max_lr = 1e-4
        total_steps = len(train_loader) * n_epochs
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
        )

        # Training loop
        for epoch in range(n_epochs):
            losses = []
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # apply direct transform
                X_batch = direct_transform(X_batch)

                with torch.cuda.amp.autocast():
                    output = self.model(x_enc=X_batch)
                    loss = criterion(inverse_transform(output.forecast), y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                # scheduler.step()

                losses.append(loss.item())

            avg_loss = np.mean(losses)

            # update lr using scheduler
            scheduler.step()

            # Validation
            val_losses = []
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                with torch.no_grad():
                    output = self.model(x_enc=X_val)
                    val_loss = criterion(output.forecast, y_val)

                val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            # Early stopping based on validation loss
            if epoch == 0:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                else:
                    patience_counter += 1

            if patience_counter >= max_patience:  # Early stopping patience
                if logger:
                    logger.info(f"Early stopping at epoch {epoch}")
                else:
                    print(f"Early stopping at epoch {epoch}")
                break

            if verbose:
                if logger:
                    logger.info(
                        f"Epoch {epoch}: Train loss: {avg_loss:.3f}, Val loss: "
                        f"{avg_val_loss:.3f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}: Train loss: {avg_loss:.3f}, Val loss: "
                        f"{avg_val_loss:.3f}"
                    )

        # Restore the best model weights
        if logger:
            logger.info(f"Restoring weights from epoch {best_epoch}")
        else:
            print(f"Restoring weights from epoch {best_epoch}")
        self.model.load_state_dict(best_model_weights)
        del best_model_weights
