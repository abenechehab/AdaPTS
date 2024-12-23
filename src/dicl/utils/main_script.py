import os
import csv
import logging
from datetime import datetime

# dicl
from dicl.utils import data_readers

# Moment
from momentfm import MOMENTPipeline

# Moirai
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


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

    return root
