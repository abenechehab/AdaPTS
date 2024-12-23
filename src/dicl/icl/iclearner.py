from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Optional, List
from dataclasses import dataclass

import copy
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import torch
# from torch.optim.lr_scheduler import OneCycleLR

from dicl.utils.icl import (
    serialize_arr,
    SerializerSettings,
    calculate_multiPDF_llama3,
)

from momentfm.utils.utils import control_randomness

if TYPE_CHECKING:
    from transformers import AutoModel, AutoTokenizer
    from dicl.utils.icl import MultiResolutionPDF
    from momentfm import MOMENTPipeline
    from uni2ts.model.moirai import MoiraiForecast


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
    def predict_long_horizon(self, prediction_horizon: int, **kwargs):
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

    def predict_long_horizon(
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
        time_series: NDArray[np.float32],
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
            self.icl_object[dim].mean_arr = preds
            self.icl_object[dim].mode_arr = preds
            self.icl_object[dim].sigma_arr = np.zeros_like(preds)
        return self.icl_object

    def predict_long_horizon(
        self,
        prediction_horizon: int,
        batch_size: int = 512,
        native_multivariate: bool = False,
        verbose: int = 1,
    ):
        """Multi-step prediction using MOMENT model"""
        self.model.eval()
        # Get device from model
        device = next(self.model.parameters()).device
        if native_multivariate:
            # Process all features together
            tensor_ts = torch.cat(
                [
                    torch.from_numpy(self.icl_object[dim].time_series)
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
                batch_predictions = (
                    self.model(x_enc=batch_ts).forecast.cpu().detach().numpy()
                )
                all_predictions.append(batch_predictions)

            # Stack all batches together
            predictions = np.concatenate(all_predictions, axis=0)

            for dim in range(self.n_features):
                self.icl_object[dim].predictions = np.expand_dims(
                    predictions[:, dim, :], axis=1
                )
        else:
            for dim in tqdm(range(self.n_features), desc="feature"):
                ts = self.icl_object[dim].time_series
                tensor_ts = torch.from_numpy(ts).float().to(device)
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
                    batch_predictions = (
                        self.model(x_enc=batch_ts).forecast.cpu().detach().numpy()
                    )
                    all_predictions.append(batch_predictions)
                # Stack all batches together
                predictions = np.concatenate(all_predictions, axis=0)
                self.icl_object[dim].predictions = predictions
        return self.compute_statistics()

    def fine_tune(
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

        self.model.train()

        # Set random seeds for PyTorch, Numpy etc.
        control_randomness(seed=seed)

        # Get device from model
        device = next(self.model.parameters()).device

        # Create dataset and data loader
        tensor_X = torch.from_numpy(X).float()
        tensor_y = torch.from_numpy(y).float()
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Setup training
        criterion = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()

        # Create a OneCycleLR scheduler
        # max_lr = 1e-4
        # total_steps = len(train_loader) * n_epochs
        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=max_lr,
        #     total_steps=total_steps,
        #     pct_start=0.3
        # )

        # Training loop
        for epoch in range(n_epochs):
            losses = []
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                with torch.cuda.amp.autocast():
                    output = self.model(x_enc=X_batch)
                    loss = criterion(output.forecast, y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                # scheduler.step()

                losses.append(loss.item())

            avg_loss = np.mean(losses)
            if verbose:
                print(f"Epoch {epoch}: Train loss: {avg_loss:.3f}")


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
        return self.icl_object

    def predict_long_horizon(
        self,
        prediction_horizon: int,
        batch_size: int = 1024,
    ):
        """Multi-step prediction using Moirai model"""
        self.model.eval()
        # Get device from model
        device = next(self.model.parameters()).device
        for dim in range(self.n_features):
            ts = self.icl_object[dim].time_series
            tensor_ts = torch.from_numpy(ts).float().to(device)
            # Time series values. Shape: (batch, time, variate)
            tensor_ts = tensor_ts.reshape((self.batch_size, self.context_length, 1))
            # Process in batches to avoid memory issues
            all_predictions = []
            for i in tqdm(range(0, self.batch_size, batch_size), desc="batch"):
                batch_end = min(i + batch_size, self.batch_size)
                batch_ts = tensor_ts[i:batch_end]

                batch_predictions = (
                    self.model(
                        past_target=batch_ts,
                        past_observed_target=torch.ones_like(
                            batch_ts, dtype=torch.bool
                        ),
                        past_is_pad=torch.zeros_like(
                            batch_ts, dtype=torch.bool
                        ).squeeze(-1),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                all_predictions.append(batch_predictions)

            # Stack all batches together
            predictions = np.concatenate(all_predictions, axis=0)

            # expand_dims to make it (batch, variate=1, time)
            self.icl_object[dim].predictions = np.expand_dims(
                np.round(np.mean(predictions, axis=1), decimals=4), axis=1
            )
            self.icl_object[dim].mean_arr = np.expand_dims(
                np.round(np.mean(predictions, axis=1), decimals=4), axis=1
            )
            self.icl_object[dim].mode_arr = copy.copy(self.icl_object[dim].mean_arr)
            self.icl_object[dim].sigma_arr = np.expand_dims(
                np.round(np.std(predictions, axis=1), decimals=4), axis=1
            )

        return self.compute_statistics()


# class TTMICLTrainer(ICLTrainer):
#     def __init__(
#         self, model: "MOMENTPipeline", n_features: int, forecast_horizon: int = 96
#     ):
#         # TODO: change type of model when Moirai is installed
#         """
#         TTMICLTrainer is an implementation of ICLTrainer using the TTM
#         foundation model for time series forecasting.

#         Args:
#             n_features (int): Number of features in the time series data
#             forecast_horizon (int): Number of steps to forecast
#             rescale_factor (float): Rescaling factor for data normalization
#             up_shift (float): Shift value applied after rescaling
#         """

#         self.model = model

#         self.n_features = n_features
#         self.forecast_horizon = forecast_horizon

#         self.icl_object: List[ICLObject] =
# [ICLObject() for _ in range(self.n_features)]

#         self.context_length = None
#         self.batch_size = None

#     def update_context(
#         self,
#         time_series: NDArray[np.float32],
#         context_length: Optional[int] = None,
#     ):
#         """Updates the context with given time series data"""
#         if context_length is not None:
#             self.context_length = context_length
#         else:
#             self.context_length = time_series.shape[-1]

#         assert len(time_series.shape) == 3 and time_series.shape[1] == self.n_features

#         self.batch_size = time_series.shape[0]

#         # Store original time series for each feature
#         for dim in range(self.n_features):
#             self.icl_object[dim].time_series = time_series[
#                 :, dim, : self.context_length
#             ]

#         return self.icl_object

#     def predict_long_horizon(
#         self,
#     ):
#         """Multi-step prediction using Moirai model"""
#         self.model.eval()
#         # Get device from model
#         device = next(self.model.parameters()).device
#         for dim in range(self.n_features):
#             ts = self.icl_object[dim].time_series
#             tensor_ts = torch.from_numpy(ts).float().to(device)
#             # Time series values. Shape: (batch, time, variate)
#             tensor_ts = tensor_ts.reshape((self.batch_size, self.context_length, 1))

#             temp_dir = tempfile.mkdtemp()
#             # zeroshot_trainer
#             zeroshot_trainer = Trainer(
#                 model=zeroshot_model,
#                 args=TrainingArguments(
#                     output_dir=temp_dir,
#                     per_device_eval_batch_size=batch_size,
#                     seed=SEED,
#                     report_to="none",
#                 ),
#             )
#             # evaluate = zero-shot performance
#             print("+" * 20, "Test MSE zero-shot", "+" * 20)
#             zeroshot_output = zeroshot_trainer.evaluate(dset_test)
#             print(zeroshot_output)

#             # get predictions

#             predictions_dict = zeroshot_trainer.predict(dset_test)

#             predictions_np = predictions_dict.predictions[0]

#             print(predictions_np.shape)

#             self.icl_object[dim].predictions = (
#                 np.round(np.median(predictions[0], axis=0), decimals=4),
#             )
#             self.icl_object[dim].mean_arr = (
#                 np.round(np.mean(predictions[0], axis=0), decimals=4),
#             )
#             self.icl_object[dim].mode_arr = (
#                 np.round(np.mean(predictions[0], axis=0), decimals=4),
#             )
#             self.icl_object[dim].sigma_arr = (
#                 np.round(np.std(predictions[0], axis=0), decimals=4),
#             )
#         return self.icl_object
