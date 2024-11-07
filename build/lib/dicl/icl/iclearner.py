from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Optional, List
from dataclasses import dataclass

import copy
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray

from dicl.utils.icl import (
    serialize_arr,
    SerializerSettings,
    calculate_multiPDF_llama3,
)

if TYPE_CHECKING:
    from transformers import AutoModel, AutoTokenizer
    from dicl.utils.icl import MultiResolutionPDF

from moment import MOMENTPipeline


@dataclass
class ICLObject:
    time_series: Optional[NDArray[np.float32]] = None
    mean_series: Optional[NDArray[np.float32]] = None
    sigma_series: Optional[NDArray[np.float32]] = None
    str_series: Optional[str] = None
    rescaled_true_mean_arr: Optional[NDArray[np.float32]] = None
    rescaled_true_sigma_arr: Optional[NDArray[np.float32]] = None
    rescaling_min: Optional[NDArray[np.float32]] = None
    rescaling_max: Optional[NDArray[np.float32]] = None
    PDF_list: Optional[List] = None
    predictions: Optional[NDArray[np.float32]] = None
    mean_arr: Optional[NDArray[np.float32]] = None
    mode_arr: Optional[NDArray[np.float32]] = None
    sigma_arr: Optional[NDArray[np.float32]] = None


class ICLTrainer(ABC):
    """ICLTrainer that takes a time serie and processes it using the LLM."""

    @abstractmethod
    def update_context(self, time_series: NDArray[np.float32], **kwargs) -> ICLObject:
        """Update the context (internal state) with the given time serie."""

    @abstractmethod
    def compute_statistics(self, **kwargs) -> ICLObject:
        """Compute useful statistics for the predicted PDFs in the internal state."""

    @abstractmethod
    def predict_long_horizon(self, **kwargs):
        """Long horizon autoregressive predictions using the model."""


class MultiVariateICLTrainer(ICLTrainer):
    def __init__(
        self,
        model: "AutoModel",
        tokenizer: "AutoTokenizer",
        n_features: int,
        rescale_factor: float = 7.0,
        up_shift: float = 1.5,
    ):
        """
        MultiVariateICLTrainer is an implementation of ICLTrainer for multivariate time
        series data.
        It uses an LLM to process time series and make predictions.

        Args:
            model (AutoModel): The LLM model used for ICL.
            tokenizer (AutoTokenizer): Tokenizer associated with the model.
            n_features (int): Number of features in the time series data.
            rescale_factor (float, optional): Rescaling factor for data normalization.
                Default is 7.0.
            up_shift (float, optional): Shift value applied after rescaling.
                Default is 1.5.
        """
        self.model: "AutoModel" = model
        self.tokenizer: "AutoTokenizer" = tokenizer

        self.n_features: int = n_features

        self.use_cache: bool = False

        self.up_shift: float = up_shift
        self.rescale_factor: float = rescale_factor

        self.icl_object: List[ICLObject] = [ICLObject() for _ in range(self.n_features)]
        self.kv_cache: List[Optional[NDArray[np.float32]]] = [
            None for _ in range(self.n_features)
        ]

    def update_context(
        self,
        time_series: NDArray[np.float32],
        mean_series: NDArray[np.float32],
        sigma_series: NDArray[np.float32],
        context_length: Optional[int] = None,
        update_min_max: bool = True,
    ):
        """
        Updates the context (internal state) with the given time series data.

        Args:
            time_series (NDArray[np.float32]): Input time series data.
            mean_series (NDArray[np.float32]): Mean of the time series data.
            sigma_series (NDArray[np.float32]): Standard deviation of the time series
                data. (only relevant if the stochastic data generation process is known)
            context_length (Optional[int], optional): The length of the time series.
                If None, the full time series length is used.
            update_min_max (bool, optional): Whether to update the minimum and maximum
                rescaling values. Default is True.

        Returns:
            List[ICLObject]: A list of ICLObject instances representing the updated
                internal state for each feature.
        """
        if context_length is not None:
            self.context_length = context_length
        else:
            self.context_length = time_series.shape[0]
        assert (
            len(time_series.shape) > 1 and time_series.shape[1] == self.n_features
        ), f"Not all features ({self.n_features}) are given in time series of shape: "
        f"{time_series.shape}"

        for dim in range(self.n_features):
            # ------------------ serialize_gaussian ------------------
            settings = SerializerSettings(
                base=10,
                prec=2,
                signed=True,
                time_sep=",",
                bit_sep="",
                minus_sign="-",
                fixed_length=False,
                max_val=10,
            )

            if update_min_max:
                self.icl_object[dim].rescaling_min = time_series[
                    : self.context_length, dim
                ].min()
                self.icl_object[dim].rescaling_max = time_series[
                    : self.context_length, dim
                ].max()

            ts_min = copy.copy(self.icl_object[dim].rescaling_min)
            ts_max = copy.copy(self.icl_object[dim].rescaling_max)
            rescaled_array = (time_series[: self.context_length, dim] - ts_min) / (
                ts_max - ts_min
            ) * self.rescale_factor + self.up_shift
            rescaled_true_mean_arr = (
                mean_series[: self.context_length, dim] - ts_min
            ) / (ts_max - ts_min) * self.rescale_factor + self.up_shift
            rescaled_true_sigma_arr = (
                sigma_series[: self.context_length, dim]
                / (ts_max - ts_min)
                * self.rescale_factor
            )

            full_series = serialize_arr(rescaled_array, settings)

            self.icl_object[dim].time_series = time_series[: self.context_length, dim]
            self.icl_object[dim].mean_series = mean_series[: self.context_length, dim]
            self.icl_object[dim].sigma_series = sigma_series[: self.context_length, dim]
            self.icl_object[dim].rescaled_true_mean_arr = rescaled_true_mean_arr
            self.icl_object[dim].rescaled_true_sigma_arr = rescaled_true_sigma_arr
            self.icl_object[dim].str_series = full_series
        return self.icl_object

    def icl(
        self,
        temperature: float = 1.0,
        n_states: int = 1000,
        stochastic: bool = False,
        use_cache: bool = False,
        verbose: int = 0,
        if_true_mean_else_mode: bool = False,
    ):
        """
        Performs In-Context Learning (ICL) using the LLM for multivariate time series.

        Args:
            temperature (float, optional): Sampling temperature for predictions.
                Default is 1.0.
            n_states (int, optional): Number of possible states for the PDF prediction.
                Default is 1000.
            stochastic (bool, optional): If True, stochastic sampling is used for
                predictions. Default is False.
            use_cache (bool, optional): If True, uses cached key values to improve
                efficiency. Default is False.
            verbose (int, optional): Verbosity level for progress tracking.
                Default is 0.
            if_true_mean_else_mode (bool, optional): Whether to use the true mean or
                mode for prediction (only relevant if stochastic=False).
                Default is False.

        Returns:
            List[ICLObject]: A list of ICLObject instances with updated PDFs and
                predictions for each feature.
        """
        self.use_cache = use_cache
        for dim in tqdm(
            range(self.n_features), desc="icl / state dim", disable=not bool(verbose)
        ):
            PDF_list, _, kv_cache = calculate_multiPDF_llama3(
                self.icl_object[dim].str_series,
                model=self.model,
                tokenizer=self.tokenizer,
                n_states=n_states,
                temperature=temperature,
                use_cache=self.use_cache,
            )
            self.kv_cache[dim] = kv_cache

            self.icl_object[dim].PDF_list = PDF_list

            ts_min = self.icl_object[dim].rescaling_min
            ts_max = self.icl_object[dim].rescaling_max

            predictions = []
            for timestep in range(len(PDF_list)):
                PDF: "MultiResolutionPDF" = PDF_list[timestep]
                PDF.compute_stats()

                # Calculate the mode of the PDF
                if stochastic:
                    raw_state = np.random.choice(
                        PDF.bin_center_arr,
                        p=PDF.bin_height_arr / np.sum(PDF.bin_height_arr),
                    )
                else:
                    raw_state = PDF.mean if if_true_mean_else_mode else PDF.mode
                next_state = ((raw_state - self.up_shift) / self.rescale_factor) * (
                    ts_max - ts_min
                ) + ts_min
                predictions.append(next_state)

            self.icl_object[dim].predictions = np.array(predictions)

        return self.icl_object

    def compute_statistics(
        self,
    ):
        """
        Computes statistics (mean, mode, and sigma) for the predicted PDFs in the
            internal state.

        Returns:
            List[ICLObject]: A list of ICLObject instances with computed statistics for
                each feature.
        """
        for dim in range(self.n_features):
            PDF_list = self.icl_object[dim].PDF_list

            PDF_true_list = copy.deepcopy(PDF_list)

            ### Extract statistics from MultiResolutionPDF
            mean_arr = []
            mode_arr = []
            sigma_arr = []
            for PDF, PDF_true, true_mean, true_sigma in zip(
                PDF_list,
                PDF_true_list,
                self.icl_object[dim].rescaled_true_mean_arr,
                self.icl_object[dim].rescaled_true_sigma_arr,
            ):
                PDF.compute_stats()
                mean, mode, sigma = PDF.mean, PDF.mode, PDF.sigma

                mean_arr.append(mean)
                mode_arr.append(mode)
                sigma_arr.append(sigma)

            self.icl_object[dim].mean_arr = np.array(mean_arr)
            self.icl_object[dim].mode_arr = np.array(mode_arr)
            self.icl_object[dim].sigma_arr = np.array(sigma_arr)
        return self.icl_object

    def predict_long_horizon_llm(
        self,
        prediction_horizon: int,
        temperature: float = 1.0,
        stochastic: bool = False,
        verbose: int = 0,
        if_true_mean_else_mode: bool = False,
    ):
        """
        Predicts multiple steps into the future by autoregressively using previous
        predictions.

        Args:
            prediction_horizon (int): The number of future steps to predict.
            temperature (float, optional): Sampling temperature for predictions.
                Default is 1.0.
            stochastic (bool, optional): If True, stochastic sampling is used for
                predictions. Default is False.
            verbose (int, optional): Verbosity level for progress tracking.
                Default is 0.
            if_true_mean_else_mode (bool, optional): Whether to use the true mean or
                mode for predictions (only relevant if stochastic=False).
                Default is False.

        Returns:
            List[ICLObject]: A list of ICLObject instances with the predicted time
                series and computed statistics.
        """
        last_prediction = copy.copy(
            np.concatenate(
                [
                    self.icl_object[dim].predictions[-1].reshape((1, 1))
                    for dim in range(self.n_features)
                ],
                axis=1,
            )
        )

        current_ts = copy.copy(
            np.concatenate(
                [
                    self.icl_object[dim].time_series.reshape((-1, 1))
                    for dim in range(self.n_features)
                ],
                axis=1,
            )
        )
        for h in tqdm(
            range(prediction_horizon),
            desc="prediction_horizon",
            disable=not bool(verbose),
        ):
            input_time_series = np.concatenate([current_ts, last_prediction], axis=0)

            self.update_context(
                time_series=input_time_series,
                mean_series=copy.copy(input_time_series),
                sigma_series=np.zeros_like(input_time_series),
                context_length=self.context_length + h + 1,
                update_min_max=False,  # if False, predictions get out of bounds
            )
            self.icl(
                temperature=temperature,
                stochastic=stochastic,
                if_true_mean_else_mode=if_true_mean_else_mode,
                verbose=0,
            )

            current_ts = np.concatenate([current_ts, last_prediction], axis=0)

            last_prediction = copy.copy(
                np.concatenate(
                    [
                        self.icl_object[dim].predictions[-1].reshape((1, 1))
                        for dim in range(self.n_features)
                    ],
                    axis=1,
                )
            )

        return self.compute_statistics()


class MomentICLTrainer(ICLTrainer):
    def __init__(
        self,
        n_features: int,
        forecast_horizon: int = 96,
        rescale_factor: float = 7.0,
        up_shift: float = 1.5,
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

        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": forecast_horizon,
            },
        )
        self.model.init()

        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.up_shift = up_shift
        self.rescale_factor = rescale_factor

        self.icl_object: List[ICLObject] = [ICLObject() for _ in range(self.n_features)]

    def update_context(
        self,
        time_series: NDArray[np.float32],
        mean_series: Optional[NDArray[np.float32]] = None,
        sigma_series: Optional[NDArray[np.float32]] = None,
        context_length: Optional[int] = None,
        update_min_max: bool = True,
    ):
        """Updates the context with given time series data"""
        if context_length is not None:
            self.context_length = context_length
        else:
            self.context_length = time_series.shape[0]

        assert len(time_series.shape) > 1 and time_series.shape[1] == self.n_features

        # Store original time series for each feature
        for dim in range(self.n_features):
            self.icl_object[dim].time_series = time_series[: self.context_length, dim]

            if update_min_max:
                self.icl_object[dim].rescaling_min = time_series[
                    : self.context_length, dim
                ].min()
                self.icl_object[dim].rescaling_max = time_series[
                    : self.context_length, dim
                ].max()

        return self.icl_object

    def icl(self, **kwargs):
        """Generate single-step predictions using MOMENT model"""
        for dim in range(self.n_features):
            ts = self.icl_object[dim].time_series.reshape(-1, 1)
            predictions = self.model.forecast(ts)
            self.icl_object[dim].predictions = predictions.squeeze()

        return self.icl_object

    def compute_statistics(self):
        """Compute statistics on predictions"""
        for dim in range(self.n_features):
            # MOMENT provides point estimates, so mean=mode=prediction, sigma=0
            preds = self.icl_object[dim].predictions
            self.icl_object[dim].mean_arr = preds
            self.icl_object[dim].mode_arr = preds
            self.icl_object[dim].sigma_arr = np.zeros_like(preds)

        return self.icl_object

    def predict_long_horizon_llm(self, prediction_horizon: int, **kwargs):
        """Multi-step prediction using MOMENT model"""
        for dim in range(self.n_features):
            ts = self.icl_object[dim].time_series.reshape(-1, 1)
            predictions = self.model.forecast(ts, horizon=prediction_horizon)
            self.icl_object[dim].predictions = predictions.squeeze()

        return self.compute_statistics()
