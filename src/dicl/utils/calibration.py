from typing import TYPE_CHECKING, Any, Callable

import copy

import numpy as np
from numpy.typing import NDArray

from scipy.stats import uniform
from scipy.special import kolmogorov
from scipy.stats import kstwobign

if TYPE_CHECKING:
    from dicl.icl.iclearner import ICLObject
    from matplotlib.axes import Axes


def compute_ks_metric(
    groundtruth: NDArray,
    all_predictions: NDArray,
    n_features: int,
    inverse_transform: Callable,
):
    """
    Computes the Kolmogorov-Smirnov (KS) metric between the predicted and ground truth
    values for each feature in a multivariate time series.

    Args:
        groundtruth (NDArray): The ground truth values for the time series.
        icl_object (ICLObject): The ICL object containing predicted PDFs.
        n_components (int): Number of components in the model predictions.
        n_features (int): Number of features in the time series.
        inverse_transform (Callable): A function to inverse-transform the predictions to
            the time series oiginal space.
        n_traces (int, optional): Number of trace samples to generate for each
            prediction. Default is 100.

    Returns:
        Tuple[NDArray, NDArray]:
            - `kss`: Array containing the KS metrics for each feature.
            - `ks_quantiles`: Array of KS quantiles for each feature and time step.
    """

    # all_predictions shape (n_traces, n_components, n_samples, 1, forecasting_horizon)
    n_samples = all_predictions.shape[2]
    prediction_horizon = all_predictions.shape[4]
    n_traces = all_predictions.shape[0]

    kss = np.zeros((n_features, prediction_horizon))
    ece = np.zeros((n_features, prediction_horizon))
    ks_quantiles = np.zeros((n_features, prediction_horizon, n_samples))

    gather_pred = []
    for s in range(n_traces):
        gather_pred.append(
            inverse_transform(all_predictions[s].swapaxes(0, 1).squeeze())
        )
    predictions = np.array(gather_pred).transpose(3, 1, 0, 2)

    for h in range(prediction_horizon):
        for dim in range(n_features):
            per_dim_groundtruth = groundtruth[:, dim, h].flatten()

            # Compute quantiles
            quantiles = np.sort(
                np.array(
                    [
                        g > m
                        for g, m in zip(per_dim_groundtruth, predictions[h, :, :, dim])
                    ]
                ).sum(axis=1)
            )
            quantiles = quantiles / n_traces

            # Compute KS metric
            kss[dim, h] = np.max(
                np.abs(quantiles - (np.arange(len(quantiles)) / len(quantiles)))
            )
            ece[dim, h] = np.mean(
                np.abs(quantiles - (np.arange(len(quantiles)) / len(quantiles)))
            )

            ks_quantiles[dim, h, :] = quantiles

    return kss, ece, ks_quantiles


def compute_ks_metric_gaussian(
    groundtruth: NDArray,
    icl_object: "ICLObject",
    n_components: int,
    n_features: int,
    inverse_transform: Callable,
    n_traces: int = 100,
):
    """
    Computes the Kolmogorov-Smirnov (KS) metric between the predicted and ground truth
    values for each feature in a multivariate time series.

    Args:
        groundtruth (NDArray): The ground truth values for the time series.
        icl_object (ICLObject): The ICL object containing predicted PDFs.
        n_components (int): Number of components in the model predictions.
        n_features (int): Number of features in the time series.
        inverse_transform (Callable): A function to inverse-transform the predictions to
            the time series oiginal space.
        n_traces (int, optional): Number of trace samples to generate for each
            prediction. Default is 100.

    Returns:
        Tuple[NDArray, NDArray]:
            - `kss`: Array containing the KS metrics for each feature.
            - `ks_quantiles`: Array of KS quantiles for each feature and time step.
    """

    n_samples = icl_object[0].mean_arr.shape[0]
    prediction_horizon = icl_object[0].mean_arr.shape[-1]
    kss = np.zeros((n_features, prediction_horizon))
    ece = np.zeros((n_features, prediction_horizon))
    ks_quantiles = np.zeros((n_features, prediction_horizon, n_samples))
    predictions = np.empty((n_traces, n_samples, n_components, prediction_horizon))
    for h in range(prediction_horizon):
        for dim in range(n_components):
            mean = icl_object[dim].mean_arr[:, 0, h]
            std = icl_object[dim].sigma_arr[:, 0, h]
            for t in range(n_samples):
                samples = np.random.normal(mean[t], std[t], size=(n_traces,))
                predictions[:, t, dim, h] = copy.copy(samples)

    gather_pred = []
    for s in range(n_traces):
        gather_pred.append(inverse_transform(predictions[s]))
    predictions = np.array(gather_pred).transpose(3, 1, 0, 2)

    for h in range(prediction_horizon):
        for dim in range(n_features):
            per_dim_groundtruth = groundtruth[:, dim, h].flatten()

            # Compute quantiles
            quantiles = np.sort(
                np.array(
                    [
                        g > m
                        for g, m in zip(per_dim_groundtruth, predictions[h, :, :, dim])
                    ]
                ).sum(axis=1)
            )
            quantiles = quantiles / n_traces

            # Compute KS metric
            kss[dim, h] = np.max(
                np.abs(quantiles - (np.arange(len(quantiles)) / len(quantiles)))
            )
            ece[dim, h] = np.mean(
                np.abs(quantiles - (np.arange(len(quantiles)) / len(quantiles)))
            )

            ks_quantiles[dim, h, :] = quantiles

    return kss, ece, ks_quantiles


def ks_cdf(
    ks_quantiles: NDArray,
    dim: int,
    ax: "Axes",
    verbose: int = 0,
    color: Any = "b",
    pot_cdf_uniform: bool = True,
    label: str = "",
):
    """
    Plots the cumulative distribution function (CDF) of the KS quantiles for a given
        dimension, and compares it with the uniform CDF.

    Args:
        ks_quantiles (NDArray): Array of KS quantiles computed for each feature and
            time step.
        dim (int): The dimension (feature) for which to plot the CDF.
        ax (Axes): The Matplotlib Axes object on which to plot the CDF.
        verbose (int, optional): Verbosity level. If greater than 0, additional
            information will be printed. Default is 0.
        color (Any, optional): Color of the CDF plot. Default is "b" (blue).
        pot_cdf_uniform (bool, optional): Whether to plot the uniform CDF for
            comparison. Default is True.
        label (str, optional): Label for the plot. Default is an empty string.

    Returns:
        None. The CDF plot is drawn on the provided Axes object.
    """
    quantiles = ks_quantiles[dim]

    x = quantiles
    n = len(quantiles)

    target = uniform(loc=0, scale=np.max(x))  # Uniform over [0, 1]
    cdfs = target.cdf(x)
    ecdfs = np.arange(n + 1, dtype=float) / n
    gaps = np.column_stack([cdfs - ecdfs[:n], ecdfs[1:] - cdfs])

    if verbose:
        Dn = np.max(gaps)
        Kn = np.sqrt(n) * Dn
        print("Dn=%f, sqrt(n)*Dn=%f" % (Dn, Kn))
        print(
            chr(10).join(
                [
                    "For a sample of size n drawn from a uniform distribution:",
                    " the approximate Kolmogorov probability that sqrt(n)*Dn>=%f is %f"
                    % (Kn, kolmogorov(Kn)),
                    " the approximate Kolmogorov probability that sqrt(n)*Dn<=%f is %f"
                    % (Kn, kstwobign.cdf(Kn)),
                ]
            )
        )

    m_label = label  # + f" - ks = {ks:.3f}"
    ax.step(
        np.concatenate([[0], x]), ecdfs, where="post", color=color, label=m_label, lw=2
    )
    if pot_cdf_uniform:
        ax.plot(
            np.concatenate([[0], x]), np.concatenate([[0], cdfs]), "--", color="red"
        )
    ax.set_ylim([0, 1])

    # Add vertical lines marking Dn+ and Dn-
    iminus, iplus = np.argmax(gaps, axis=0)
    if np.abs(ecdfs[iminus] - cdfs[iminus]) >= np.abs(cdfs[iplus] - ecdfs[iplus + 1]):
        ax.vlines([x[iminus]], ecdfs[iminus], cdfs[iminus], color=color, lw=3)
    else:
        ax.vlines([x[iplus]], cdfs[iplus], ecdfs[iplus + 1], color=color, lw=3)
