from typing import TYPE_CHECKING, Optional, List

from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import torch

if TYPE_CHECKING:
    from uni2ts.model.moirai import MoiraiForecast

from adapts.icl.iclearner import ICLTrainer, ICLObject


class MoiraiICLTrainer(ICLTrainer):
    def __init__(
        self, model: "MoiraiForecast", n_features: int, forecast_horizon: int = 96
    ):
        """
        MoiraiICLTrainer is an implementation of ICLTrainer using the Moirai
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

        self.context_length = None
        self.batch_size = None

    def update_context(
        self,
        time_series: NDArray[np.float32],
        context_length: Optional[int] = None,
    ):
        """Updates the context with given time series data"""
        if context_length is not None:
            self.context_length = context_length
        else:
            self.context_length = time_series.shape[-1]

        assert len(time_series.shape) == 3 and time_series.shape[1] == self.n_features

        self.batch_size = time_series.shape[0]

        # Store original time series for each feature
        for dim in range(self.n_features):
            self.icl_object[dim].time_series = time_series[
                :, dim, : self.context_length
            ]

        return self.icl_object

    def compute_statistics(self):
        """Compute statistics on predictions"""
        for dim in range(self.n_features):
            # MOIRAI provides multiple samples in dim=1
            preds = self.icl_object[dim].predictions
            mean_preds = preds.mean(axis=1).unsqueeze(1).cpu().detach().numpy()
            self.icl_object[dim].mean_arr = mean_preds
            # TODO: set mode here
            self.icl_object[dim].mode_arr = mean_preds
            self.icl_object[dim].sigma_arr = (
                preds.mean(axis=1).unsqueeze(1).cpu().detach().numpy().std(axis=1)
            )
            self.icl_object[dim].predictions = preds.median(axis=1).values.unsqueeze(1)
        return self.icl_object

    def predict_long_horizon(
        self,
        prediction_horizon: int,
        batch_size: int = 1024,
        native_multivariate: bool = True,
        verbose: int = 1,
    ):
        """Multi-step prediction using Moirai model"""
        self.model.eval()
        # Get device from model
        device = next(self.model.parameters()).device
        if native_multivariate:
            # Process all features together. Shape: (batch, time, variate)
            tensor_ts = torch.cat(
                [
                    self.icl_object[dim].time_series.unsqueeze(-1).float().to(device)
                    if isinstance(self.icl_object[dim].time_series, torch.Tensor)
                    else torch.from_numpy(self.icl_object[dim].time_series)
                    .unsqueeze(-1)
                    .float()
                    .to(device)
                    for dim in range(self.n_features)
                ],
                axis=-1,
            )
            # Process in batches to avoid memory issues
            all_predictions = []
            for i in tqdm(
                range(0, self.batch_size, batch_size),
                desc="inference batch",
                disable=not bool(verbose),
            ):
                batch_end = min(i + batch_size, self.batch_size)
                batch_ts = tensor_ts[i:batch_end]
                batch_predictions = self.model(
                    past_target=batch_ts,
                    past_observed_target=torch.ones_like(batch_ts, dtype=torch.bool),
                    past_is_pad=torch.zeros_like(batch_ts, dtype=torch.bool)[:, :, 0],
                )
                all_predictions.append(batch_predictions)

            # Stack all batches together
            predictions = torch.concatenate(all_predictions, axis=0)

            for dim in range(self.n_features):
                self.icl_object[dim].predictions = predictions[:, :, :, dim]
        else:
            for dim in range(self.n_features):
                ts = self.icl_object[dim].time_series
                tensor_ts = ts if isinstance(ts, torch.Tensor) else torch.from_numpy(ts)
                tensor_ts = tensor_ts.float().to(device)
                # Time series values. Shape: (batch, time, variate)
                tensor_ts = tensor_ts.reshape((self.batch_size, self.context_length, 1))
                # Process in batches to avoid memory issues
                all_predictions = []
                for i in tqdm(
                    range(0, self.batch_size, batch_size),
                    desc="inference batch",
                    disable=not bool(verbose),
                ):
                    batch_end = min(i + batch_size, self.batch_size)
                    batch_ts = tensor_ts[i:batch_end]
                    batch_predictions = self.model(
                        past_target=batch_ts,
                        past_observed_target=torch.ones_like(
                            batch_ts, dtype=torch.bool
                        ),
                        past_is_pad=torch.zeros_like(
                            batch_ts, dtype=torch.bool
                        ).squeeze(-1),
                    )
                    all_predictions.append(batch_predictions)
                predictions = torch.concat(all_predictions, axis=0)
                self.icl_object[dim].predictions = predictions
        return self.compute_statistics()
