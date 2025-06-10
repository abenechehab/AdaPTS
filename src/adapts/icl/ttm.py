from typing import Optional, List

import numpy as np
from numpy.typing import NDArray
import torch

from adapts.icl.iclearner import ICLTrainer, ICLObject

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.get_model import get_model


def load_ttm_model(
    model_name: str,
    forecast_horizon: int,
    context_length: int,
) -> TinyTimeMixerForPrediction:
    zeroshot_model = get_model(
        model_name,
        context_length=context_length,
        prediction_length=max(forecast_horizon, 96),
        freq_prefix_tuning=False,
        freq=None,
        prefer_l1_loss=False,
        prefer_longer_context=True,
    )
    return zeroshot_model


class TTMICLTrainer(ICLTrainer):
    def __init__(
        self,
        model: "TinyTimeMixerForPrediction",
        n_features: int,
        forecast_horizon: int = 96,
    ):
        # TODO: change type of model when Moirai is installed
        """
        TTMICLTrainer is an implementation of ICLTrainer using the TTM
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

    def predict_long_horizon(
        self,
        prediction_horizon: int,
        batch_size: int = 256,
        seed: int = 7,
        verbose: int = 0,
    ):
        """Multi-step prediction using Moirai model"""
        self.model.eval()
        # Get device from model
        device = next(self.model.parameters()).device
        for dim in range(self.n_features):
            ts = self.icl_object[dim].time_series
            tensor_ts = ts if isinstance(ts, torch.Tensor) else torch.from_numpy(ts)
            tensor_ts = tensor_ts.float().to(device)
            # Time series values. Shape: (batch, time, variate)
            tensor_ts = tensor_ts.reshape((self.batch_size, self.context_length, 1))

            predictions = self.model(tensor_ts)
            predictions = predictions.prediction_outputs.swapaxes(1, 2)

            # trim predictions to the forecast horizon
            predictions = predictions[:, :, : self.forecast_horizon]

            self.icl_object[dim].predictions = predictions

        return self.compute_statistics()

    def compute_statistics(self):
        """Compute statistics on predictions"""
        for dim in range(self.n_features):
            # MOIRAI provides multiple samples in dim=1
            preds = self.icl_object[dim].predictions
            mean_preds = preds.cpu().detach().numpy()
            self.icl_object[dim].mean_arr = mean_preds
            # TODO: set mode here
            self.icl_object[dim].mode_arr = mean_preds
            self.icl_object[dim].sigma_arr = np.zeros_like(mean_preds)
        return self.icl_object

    def eval(self):
        self.model.eval()
        return self.model
