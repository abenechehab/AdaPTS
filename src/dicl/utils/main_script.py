import os
import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# dicl
from dicl.utils import data_readers

# Moment
from momentfm import MOMENTPipeline

# Moirai
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


RL_DATASETS = ["HalfCheetah_expert"]


def load_moment_model(model_name: str, forecast_horizon: int) -> MOMENTPipeline:
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": forecast_horizon,
            "head_dropout": 0.1,
            "weight_decay": 0,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
        local_files_only=True,
    )
    model.init()
    return model


def load_moirai_model(
    model_name: str,
    forecast_horizon: int,
    context_length: int,
) -> MoiraiForecast:
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(
            model_name,
        ),
        prediction_length=forecast_horizon,
        context_length=context_length,
        patch_size=32,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    return model


def prepare_data(dataset_name: str, context_length: int):
    datareader = data_readers.DataReader(
        data_path="/mnt/vdb/abenechehab/dicl-adapters/external_data/",
        transform_ts_size=context_length,
        univariate=False,
    )

    X_train, y_train = datareader.read_dataset(
        dataset_name=dataset_name, training_set=True
    )
    X_test, y_test = datareader.read_dataset(
        dataset_name=dataset_name, training_set=False
    )

    n_features = X_train.shape[1]

    return X_train, y_train, X_test, y_test, n_features


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
        for j in range(end_idx - start_idx - context_length - forecasting_horizon + 1):
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
    elapsed_time,
    seed,
):
    columns = [
        "dataset",
        "foundational_model",
        "adapter",
        "n_features",
        "n_components",
        "is_fine_tuned",
        "context_length",
        "forecast_horizon",
        "running_time",
        "seed",
        "metric",
        "value",
    ]

    data_row = [
        dataset_name,
        model_name,
        adapter,
        n_features,
        n_components,
        is_fine_tuned,
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
            row = data_row + [metric, value]
            writer.writerow(row)


# At the start of your program, configure logging once:
def setup_logging(logger_name, log_level, log_dir) -> logging.Logger:
    # Clear existing handlers
    root = logging.getLogger(logger_name)
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir, timestamp)
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

    return root, os.path.join(log_dir, timestamp)
