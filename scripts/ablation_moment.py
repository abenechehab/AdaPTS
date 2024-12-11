import os
import csv
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Optional
import tyro
import numpy as np
import torch

# DICL
from dicl import dicl, adapters
from dicl.icl import iclearner as icl
from dicl.utils import data_readers

# Moment
from momentfm import MOMENTPipeline

# Moirai
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


os.environ["HF_HOME"] = "/mnt/vdb/hugguingface/"


@dataclass
class Args:
    is_fine_tuned: bool = False
    forecast_horizon: int = 96
    model_name: str = "AutonLab/MOMENT-1-large"  # f"Salesforce/moirai-1.1-R-large"
    context_length: int = 512
    dataset_name: str = "ETTh1"  # Will be set based on forecast_horizon
    adapter: Optional[str] = None  # "pca"
    data_path: Path = Path("/mnt/vdb/abenechehab/dicl-adapters/results/data.csv")
    seed: int = 13
    device: str = "cpu"


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

    batch_size = -1
    time_series = np.concatenate([X_test, y_test], axis=-1)

    if batch_size > 0:
        time_series = time_series[:batch_size]

    n_features = time_series.shape[1]

    return time_series, X_train, y_train, X_test, y_test, n_features


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


def main(args: Args):
    # Set dataset name based on forecast horizon if not provided
    dataset_name = f"{args.dataset_name}_pred={args.forecast_horizon}"

    time_series, X_train, y_train, _, _, n_features = prepare_data(
        dataset_name, args.context_length
    )

    start = n_features if not args.adapter else 1
    end = n_features + 1

    for n_components in range(start, end):
        start_time = time.time()

        disentangler = adapters.MultichannelProjector(
            num_channels=n_features,
            new_num_channels=n_components,
            patch_window_size=None,
            base_projector=args.adapter,
        )

        if "MOMENT" in args.model_name:
            model = load_moment_model(args.model_name, args.forecast_horizon).to(
                torch.device(args.device)
            )
            iclearner = icl.MomentICLTrainer(
                model=model,
                n_features=n_components,
                forecast_horizon=args.forecast_horizon,
            )
        elif "moirai" in args.model_name:
            model = load_moirai_model(
                args.model_name, args.forecast_horizon, args.context_length
            ).to(torch.device(args.device))
            iclearner = icl.MoiraiICLTrainer(
                model=model,
                n_features=n_components,
                forecast_horizon=args.forecast_horizon,
            )
        elif "ttm" in args.model_name:
            raise NotImplementedError("TTM backbone not implemented yet")
        else:
            raise ValueError(f"Not supported model: {args.model_name}")

        DICL = dicl.DICL(
            disentangler=disentangler,
            iclearner=iclearner,
            n_features=n_features,
            n_components=n_components,
        )

        DICL.fit_disentangler(X=X_train)

        if args.is_fine_tuned:
            DICL.fine_tune_iclearner(
                X=X_train,
                y=y_train,
                n_epochs=1,
                batch_size=8,
                learning_rate=1e-4,
                max_grad_norm=5.0,
                verbose=1,
                seed=args.seed,
            )

        _, _, _, _ = DICL.predict_multi_step(
            X=time_series,
            prediction_horizon=args.forecast_horizon,
        )

        metrics = DICL.compute_metrics()

        save_metrics_to_csv(
            metrics,
            args.dataset_name,
            args.model_name,
            args.adapter,
            n_features,
            n_components,
            args.context_length,
            args.forecast_horizon,
            args.data_path,
            is_fine_tuned=args.is_fine_tuned,
            elapsed_time=time.time() - start_time,
            seed=args.seed,
        )

        del DICL, disentangler, iclearner, model


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
