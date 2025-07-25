import os
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# adapts
from adapts.utils import data_readers


RL_DATASETS = ["HalfCheetah_expert"]


def prepare_data(dataset_name: str, context_length: int, forecasting_horizon: int):
    if dataset_name in RL_DATASETS:
        X_train, y_train, X_test, y_test, n_features = prepare_data_rl(
            dataset_name=dataset_name,
            context_length=context_length,
            n_observations=17,
            n_actions=6,
            forecasting_horizon=forecasting_horizon,
            include_actions=True,
        )
        return X_train, y_train, None, None, X_test, y_test, n_features
    else:
        datareader = data_readers.DataReader(
            # TODO: handle this as parameter to avoid absolute path
            data_path="/mnt/data_2/abenechehab/AdaPTS/external_data/",
            transform_ts_size=context_length,
            univariate=False,
        )

        X_train, y_train = datareader.read_dataset(
            dataset_name=dataset_name, setting="train"
        )
        X_val, y_val = datareader.read_dataset(dataset_name=dataset_name, setting="val")
        X_test, y_test = datareader.read_dataset(
            dataset_name=dataset_name, setting="test"
        )

        n_features = X_train.shape[1]

        return X_train, y_train, X_val, y_val, X_test, y_test, n_features


def prepare_data_rl(
    dataset_name: str,
    context_length: int,
    n_observations: int = 17,
    n_actions: int = 6,
    forecasting_horizon: int = 96,
    include_actions: bool = True,
):
    env_name, data_label = dataset_name.split("_")[0], dataset_name.split("_")[1]
    data_label = "expert"
    data_path = Path("src") / "dicl" / "data" / f"D4RL_{env_name}_{data_label}.csv"

    # to use DICL-(s,a), set include_actions to True
    if include_actions:
        n_features = n_observations + n_actions
    else:
        n_features = n_observations

    # load data to get a sample episode
    X = pd.read_csv(data_path, index_col=0)
    X = X.values.astype("float")

    # find episodes beginnings. the restart column is equal to 1 at the start of
    # an episode, 0 otherwise.
    restart_index = n_observations + n_actions + 1
    restarts = X[:, restart_index]

    # get all episodes
    episode_indices = np.where(np.append(restarts, 1))[0]

    # Prepare lists to store training data
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    # Randomly select one episode for testing
    test_episode_idx = np.random.randint(0, len(episode_indices) - 1)

    # Process each episode
    for i in range(len(episode_indices) - 1):
        if i == test_episode_idx:
            # Save test episode
            start_idx = episode_indices[i]
            end_idx = episode_indices[i + 1]
            assert end_idx - start_idx >= context_length + forecasting_horizon, "Episo"
            "de is too short"
            for j in range(
                end_idx - start_idx - context_length - forecasting_horizon + 1
            ):
                data_window = X[
                    start_idx + j : start_idx
                    + j
                    + context_length
                    + forecasting_horizon,
                    :n_features,
                ]
                if not np.isnan(data_window).any():
                    X_test_list.append(
                        X[start_idx + j : start_idx + j + context_length, :n_features]
                    )
                    y_test_list.append(
                        X[
                            start_idx + j + context_length : start_idx
                            + j
                            + context_length
                            + forecasting_horizon,
                            :n_features,
                        ]
                    )
            continue

        # Process episode for training
        start_idx = episode_indices[i]
        end_idx = episode_indices[i + 1]

        # Skip if episode is too short
        if end_idx - start_idx < context_length + forecasting_horizon:
            continue

        # Create sliding windows
        starting_points = np.random.choice(
            range(0, end_idx - start_idx - context_length - forecasting_horizon), 10
        )
        for j in starting_points:
            data_window = X[
                start_idx + j : start_idx + j + context_length + forecasting_horizon,
                :n_features,
            ]
            if not np.isnan(data_window).any():
                X_train_list.append(
                    X[start_idx + j : start_idx + j + context_length, :n_features]
                )
                y_train_list.append(
                    X[
                        start_idx + j + context_length : start_idx
                        + j
                        + context_length
                        + forecasting_horizon,
                        :n_features,
                    ]
                )

    # Convert lists to arrays
    X_train = np.array(X_train_list).swapaxes(-1, -2)
    y_train = np.array(y_train_list).swapaxes(-1, -2)
    X_test = np.array(X_test_list).swapaxes(-1, -2)
    y_test = np.array(y_test_list).swapaxes(-1, -2)

    return X_train, y_train, X_test, y_test, n_features


def save_metrics_to_csv(
    metrics,
    dataset_name,
    model_name,
    adapter,
    n_features,
    n_components,
    context_length,
    forecast_horizon,
    data_path,
    is_fine_tuned,
    pca_in_preprocessing,
    use_revin,
    elapsed_time,
    seed,
    train_size,
):
    columns = [
        "dataset",
        "foundational_model",
        "adapter",
        "n_features",
        "n_components",
        "is_fine_tuned",
        "pca_in_preprocessing",
        "use_revin",
        "context_length",
        "forecast_horizon",
        "running_time",
        "seed",
        "metric",
        "value",
        "train_size",
    ]

    data_row = [
        dataset_name,
        model_name,
        adapter,
        n_features,
        n_components,
        is_fine_tuned,
        pca_in_preprocessing,
        use_revin,
        context_length,
        forecast_horizon,
        elapsed_time,
        seed,
    ]

    file_exists = data_path.exists()

    with open(data_path, "a" if file_exists else "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(columns)
        for metric, value in metrics.items():
            row = data_row + [metric, value] + [train_size]
            writer.writerow(row)


def save_hyperopt_metrics_to_csv(
    metrics,
    dataset_name,
    model_name,
    adapter,
    n_features,
    n_components,
    context_length,
    forecasting_horizon,
    config: dict,
    data_path: Path,
    elapsed_time: float,
    seed: int,
):
    if not data_path.exists():
        # Create the directory if it doesn't exist
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_hyperopt = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "adapter",
                "n_features",
                "n_components",
                "context_length",
                "forecasting_horizon",
                "running_time",
                "seed",
            ]
        )
    else:
        df_hyperopt = pd.read_csv(data_path)

    # Create a new row as a dictionary
    new_row = {
        "dataset": dataset_name,
        "model": model_name,
        "adapter": adapter,
        "n_features": n_features,
        "n_components": n_components,
        "context_length": context_length,
        "forecasting_horizon": forecasting_horizon,
        "running_time": elapsed_time,
        "seed": seed,
    }

    # Add metrics to the row
    for metric_name, value in metrics.items():
        new_row[metric_name] = value

    # Add config hyperparameters to the row
    for param_name, param_value in config.items():
        if "scaled" not in param_name:
            if param_name not in df_hyperopt.columns:
                df_hyperopt[param_name] = "None"
            new_row[param_name] = param_value

    # Append the new row to df_hyperopt
    df_hyperopt = pd.concat([df_hyperopt, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated dataframe
    df_hyperopt.to_csv(data_path, index=False)


# At the start of your program, configure logging once:
def setup_logging(
    logger_name: str,
    log_level: str,
    log_dir: Path,
    dataset_name: str,
    adapter: Optional[str],
    model_name: str,
) -> Tuple[logging.Logger, Path]:
    # Clear existing handlers
    root = logging.getLogger(logger_name)
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (
        log_dir / dataset_name / f"{timestamp}_{dataset_name}_{adapter}_{model_name}"
    )

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    # Set format for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Set up root logger
    root.setLevel(getattr(logging, log_level))
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return root, log_dir
