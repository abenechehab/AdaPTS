from typing import TYPE_CHECKING, Any, List, Optional, Tuple
import copy

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

from dicl.utils.preprocessing import AxisScaler

if TYPE_CHECKING:
    from dicl.icl.iclearner import ICLTrainer, ICLObject


class DICL:
    def __init__(
        self,
        disentangler: Any,
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

        self.disentangler = make_pipeline(self.scaler, disentangler)

        self.iclearner = iclearner

    def fit_disentangler(self, X: NDArray):
        """
        Fit the disentangler on the input data.

        Args:
            X (NDArray): Input time series data.
        """
        self.disentangler.fit(X)

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
        return self.disentangler.transform(X)

    def inverse_transform(self, X_transformed: NDArray) -> NDArray:
        """
        Inverse transform the data back to its original representation.

        Args:
            X_transformed (NDArray): Transformed time series data.

        Returns:
            NDArray: Data transformed back to the original space.
        """
        return self.disentangler.inverse_transform(X_transformed)

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
        metrics["mse"] = np.mean(
            (self.X[:, :, -self.prediction_horizon :] - self.mean) ** 2
        )
        # ------- MAE --------
        metrics["mae"] = np.mean(
            np.abs(self.X[:, :, -self.prediction_horizon :] - self.mean)
        )

        # ------- scaled MSE --------
        scaled_groundtruth = self.scaler.transform(
            self.X[:, :, -self.prediction_horizon :]
        )
        scaled_mean = self.scaler.transform(self.mean)
        metrics["scaled_mse"] = np.mean((scaled_groundtruth - scaled_mean) ** 2)
        metrics["scaled_mae"] = np.mean(np.abs(scaled_groundtruth - scaled_mean))

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
