import os
import time
import random
from typing import Dict, Any
from dataclasses import dataclass
import tyro
from pathlib import Path

import torch
import numpy as np

import ray
from ray import tune, train
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hebo import HEBOSearch

from dicl import adapters
from dicl.dicl import DICL
from dicl.icl import iclearner as icl
from dicl.adapters import (
    SimpleAutoEncoder,
    LinearAutoEncoder,
    VariationalAutoEncoder,
    NormalizingFlow,
)
from dicl.utils.main_script import (
    load_moment_model,
    prepare_data,
    save_hyperopt_metrics_to_csv,
)

ADAPTER_CLS = {
    "simpleAE": SimpleAutoEncoder,
    "linearAE": LinearAutoEncoder,
    "VAE": VariationalAutoEncoder,
    "flow": NormalizingFlow,
}
FULL_COMP_ADAPTERS = ["flow"]


@dataclass
class Args:
    forecasting_horizon: int = 96
    model_name: str = "AutonLab/MOMENT-1-small"
    context_length: int = 512
    dataset_name: str = "ETTh1"  # Will be set based on forecasting_horizon
    seed: int = 13
    number_n_comp_to_try: int = 4
    adapter: str = "linearAE"
    gpu_fraction_per_worker: float = 1.0
    num_samples: int = 100


def get_search_space(adapter_type: str) -> Dict[str, Any]:
    """Define search space for each adapter type"""
    base_space = {
        "learning_rate": tune.choice([1e-3, 1e-2]),
        "batch_size": tune.choice([32, 64]),
    }

    if adapter_type in ["simpleAE", "VAE"]:
        base_space.update(
            {
                "num_layers": tune.choice([1, 2]),
                "hidden_dim": tune.choice([64, 128]),
                "coeff_reconstruction": tune.choice([0.0, 1e-2, 1e-1]),
            }
        )
    elif adapter_type in ["linearAE"]:
        base_space.update(
            {
                "coeff_reconstruction": tune.choice([0.0, 1e-2, 1e-1]),
            }
        )
    elif adapter_type in ["flow"]:
        base_space.update(
            {
                "num_coupling": tune.choice([1, 2]),
                "hidden_dim": tune.choice([64, 128, 256]),
            }
        )

    return base_space


def train_adapter(
    config: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    adapter_type: str,
    n_components: int,
    forecasting_horizon: int,
    context_length: int,
    seed: int,
):
    """Training function for Ray Tune"""

    # Force GPU usage if available in the worker
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # data
    X_train, y_train, X_val, y_val, X_test, y_test, n_features = prepare_data(
        dataset_name, context_length, forecasting_horizon
    )
    time_series_val = np.concatenate([X_val, y_val], axis=-1)
    time_series_test = np.concatenate([X_test, y_test], axis=-1)

    # Configure adapter
    adapter_params = {
        "input_dim": n_features,
        "device": device,  # Use the determined device
    }
    if adapter_type not in FULL_COMP_ADAPTERS:
        adapter_params.update(
            {
                "n_components": n_components,
            }
        )
    if adapter_type in ["simpleAE", "VAE"]:
        adapter_params.update(
            {
                "num_layers": config["num_layers"],
                "hidden_dim": config["hidden_dim"],
            }
        )
    elif adapter_type in ["flow"]:
        adapter_params.update(
            {
                "num_coupling": config["num_coupling"],
                "hidden_dim": config["hidden_dim"],
            }
        )

    adapter_cls = ADAPTER_CLS[adapter_type]

    # Train
    training_params = {
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "coeff_reconstruction": config.get("coeff_reconstruction", 0),
        "early_stopping_patience": 10,
        "device": device,  # Use the determined device
        "log_dir": train.get_context().get_trial_dir(),
    }

    # model
    if "MOMENT" in model_name:
        model = load_moment_model(model_name, forecasting_horizon).to(
            device
        )  # Use the determined device
        icl_constructor = icl.MomentICLTrainer
    else:
        raise ValueError(f"Not supported model: {model_name}")
    start_time = time.time()
    # iclearner
    iclearner = icl_constructor(
        model=model,
        n_features=n_components,
        forecast_horizon=forecasting_horizon,
    )
    # adapter
    adapter = adapter_cls(**adapter_params).to(device)
    disentangler = adapters.MultichannelProjector(
        num_channels=n_features,
        new_num_channels=n_components,
        patch_window_size=None,
        base_projector=adapter,
        device=device,
    )
    # dicl
    dicl_model = DICL(
        disentangler=disentangler,
        iclearner=iclearner,
        n_features=n_features,
        n_components=n_components,
    )

    # training
    dicl_model.adapter_supervised_fine_tuning(
        X_train, y_train, X_val, y_val, **training_params
    )

    # Evaluate
    with torch.no_grad():
        _, _, _, _ = dicl_model.predict_multi_step(
            X=time_series_val,
            prediction_horizon=forecasting_horizon,
        )
        metrics = dicl_model.compute_metrics()
        _, _, _, _ = dicl_model.predict_multi_step(
            X=time_series_test,
            prediction_horizon=forecasting_horizon,
        )
        test_metrics = dicl_model.compute_metrics()

    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Save metrics to CSV
    save_hyperopt_metrics_to_csv(
        metrics=metrics,
        dataset_name=dataset_name,
        model_name=model_name,
        adapter=adapter_type,
        n_features=n_features,
        n_components=n_components,
        context_length=context_length,
        forecasting_horizon=forecasting_horizon,
        config=config,
        data_path=Path("/mnt/vdb/abenechehab/dicl-adapters/results/hyperopt.csv"),
        elapsed_time=time.time() - start_time,
        seed=seed,
    )

    return metrics


def optimize_adapter(
    model_name: str,
    adapter_type: str,
    dataset_name: str,
    n_components: int,
    forecasting_horizon: int,
    context_length: int,
    num_samples: int,
    gpu_fraction_per_worker: float,
    seed: int = 13,
):
    """Run hyperparameter optimization using Ray Tune with HEBO"""

    # Ray initialization with proper GPU configuration
    # Set default Ray results directory
    ray_results_dir = "/mnt/vdb/abenechehab/dicl-adapters/logs/ray_results"
    os.makedirs(ray_results_dir, exist_ok=True)

    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"  # List all your GPUs
            }
        },
    )

    # Define search space
    search_space = get_search_space(adapter_type)

    # Define scheduler and search algorithm
    scheduler = ASHAScheduler(max_t=300, grace_period=50, reduction_factor=2)

    # Use HEBO as the search algorithm
    search_alg = HEBOSearch(
        # space=search_space,
        metric="mse",  # Metric to optimize (ensure it's consistent with your task)
        mode="min",
    )

    # Define objective function for tuning
    def objective(config):
        return train_adapter(
            config,
            dataset_name,
            model_name,
            adapter_type,
            n_components,
            forecasting_horizon,
            context_length,
            seed,
        )

    # Set up tuner with scheduler and resources per trial
    trainable_with_gpu = tune.with_resources(
        objective, {"cpu": 1, "gpu": gpu_fraction_per_worker}
    )
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric="mse",  # Change this to the metric you care about
            mode="min",
            search_alg=search_alg,
            num_samples=num_samples,
            max_concurrent_trials=int(8 / gpu_fraction_per_worker),
            scheduler=scheduler,  # Scheduler is included here
        ),
        param_space=search_space,
        run_config=RunConfig(
            name=f"{dataset_name}_{adapter_type}_ncomp{n_components}",
            storage_path=ray_results_dir,
        ),
    )

    # Run the tuning process
    results = tuner.fit()
    best_config = results.get_best_result(metric="mse", mode="min").config

    # Shutdown Ray after tuning
    ray.shutdown()

    return best_config


# Example usage:
if __name__ == "__main__":
    args = tyro.cli(Args)

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # data
    dataset_name = f"{args.dataset_name}_pred={args.forecasting_horizon}"
    _, _, _, _, _, _, n_features = prepare_data(
        dataset_name, args.context_length, args.forecasting_horizon
    )

    # n_components
    if args.adapter in FULL_COMP_ADAPTERS:
        possible_n_components = np.array([n_features])
    else:
        possible_n_components = np.linspace(
            1, n_features, min(args.number_n_comp_to_try, n_features - 1)
        ).astype("int32")

    for n_components in possible_n_components:
        print(f"\nOptimizing {args.adapter}...")
        best_config = optimize_adapter(
            model_name=args.model_name,
            adapter_type=args.adapter,
            dataset_name=dataset_name,
            n_components=n_components,
            forecasting_horizon=args.forecasting_horizon,
            context_length=args.context_length,
            num_samples=args.num_samples,
            gpu_fraction_per_worker=args.gpu_fraction_per_worker,
            seed=args.seed,
        )
        print(f"Best config for {args.adapter} with n_comp {n_components}:")
        print(best_config)
