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

from adapts import adapters
from adapts.adapts import ADAPTS
from adapts.icl import iclearner as icl
from adapts.adapters import (
    SimpleAutoEncoder,
    LinearAutoEncoder,
    betaVAE,
    NormalizingFlow,
    AENormalizingFlow,
    betaLinearVAE,
    DropoutLinearAutoEncoder,
    LinearDecoder,
    LinearEncoder,
    likelihoodVAE,
    linearLikelihoodVAE,
)
from adapts.utils.main_script import (
    load_moment_model,
    prepare_data,
    save_hyperopt_metrics_to_csv,
)

ADAPTER_CLS = {
    "simpleAE": SimpleAutoEncoder,
    "linearAE": LinearAutoEncoder,
    "VAE": betaVAE,
    "flow": NormalizingFlow,
    "AEflow": AENormalizingFlow,
    "linearVAE": betaLinearVAE,
    "dropoutLinearAE": DropoutLinearAutoEncoder,
    "linearDecoder": LinearDecoder,
    "linearEncoder": LinearEncoder,
    "lVAE": likelihoodVAE,
    "linearlVAE": linearLikelihoodVAE,
}
NOT_FULL_COMP_ADAPTERS = []
MAX_TRAIN_SIZE = 500


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
    k_fold: int = 3
    metric: str = "mse"


def get_search_space(adapter_type: str) -> Dict[str, Any]:
    """Define search space for each adapter type"""
    base_space = {
        "learning_rate": tune.choice([1e-3]),
        "batch_size": tune.choice([32]),
        "use_revin": tune.choice([True]),
    }

    if adapter_type in ["simpleAE", "VAE", "lVAE"]:
        base_space.update(
            {
                "num_layers": tune.choice([2]),
                "hidden_dim": tune.choice([128]),
                # "coeff_reconstruction": tune.choice([0.0, 1e-2, 1e-1]),
            }
        )
    if "VAE" in adapter_type:
        base_space.update(
            {
                "beta": tune.choice([0.5, 1, 2, 4]),
            }
        )
    if adapter_type in ["flow", "AEflow"]:
        base_space.update(
            {
                "num_coupling": tune.choice([1, 2, 3]),
                "hidden_dim": tune.choice([64, 128]),
            }
        )
    if "dropout" in adapter_type or adapter_type == "AEflow":
        base_space.update(
            {
                "p": tune.choice([0.1, 0.2, 0.3, 0.4]),
            }
        )
    if "lVAE" in adapter_type:
        base_space.update(
            {
                "fixed_logvar": tune.choice([0.5, 1.0, 1.5, 2.0, 3.0, None]),
                # "fixed_logvar": tune.choice([None]),
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
    k_folds: int,
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
    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    # Limit training size
    if len(X_train) > MAX_TRAIN_SIZE:
        indices = np.random.choice(len(X_train), MAX_TRAIN_SIZE, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_size = MAX_TRAIN_SIZE
    else:
        train_size = len(X_train)

    time_series_test = np.concatenate([X_test, y_test], axis=-1)

    # Configure adapter
    adapter_params = {
        "input_dim": n_features,
        "device": device,  # Use the determined device
        "context_length": context_length,
        "forecast_horizon": forecasting_horizon,
        "use_revin": config["use_revin"],
    }
    if adapter_type != "flow":
        adapter_params.update(
            {
                "n_components": n_components,
            }
        )
    if adapter_type in ["simpleAE", "VAE", "lVAE"]:
        adapter_params.update(
            {
                "num_layers": config["num_layers"],
                "hidden_dim": config["hidden_dim"],
            }
        )
    if "flow" in adapter_type:
        adapter_params.update(
            {
                "num_coupling": config["num_coupling"],
                "hidden_dim": config["hidden_dim"],
            }
        )
    if "VAE" in adapter_type:
        adapter_params.update(
            {
                "beta": config["beta"],
            }
        )
    if "dropout" in adapter_type:
        adapter_params.update(
            {
                "p": config["p"],
            }
        )
    if "lVAE" in adapter_type:
        adapter_params.update(
            {
                "fixed_logvar": config["fixed_logvar"],
            }
        )

    adapter_cls = ADAPTER_CLS[adapter_type]

    # Train
    training_params = {
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "device": device,  # Use the determined device
        "log_dir": train.get_context().get_trial_dir(),
        "n_epochs": 100,
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

    # Implement k-fold cross validation
    fold_metrics = []

    # Create folds from training data
    dataset_size = len(X_train)
    fold_size = dataset_size // (k_folds)  # k_folds+1 for rolling k-fold

    for fold in range(k_folds):
        # Time series cross-validation
        # For fold i, train on folds [0:i] and validate on fold [i+1]

        # Rolling k-fold Cross Validation
        # train_end = (fold + 1) * fold_size
        # X_fold_train = X_train[:train_end]
        # y_fold_train = y_train[:train_end]
        # X_fold_val = X_train[train_end:]
        # y_fold_val = y_train[train_end:]

        # Standard k-fold Cross Validation
        # Calculate start and end indices for validation fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        # Create validation fold
        X_fold_val = X_train[val_start:val_end]
        y_fold_val = y_train[val_start:val_end]

        # Create training fold from remaining data
        X_fold_train = np.concatenate([X_train[:val_start], X_train[val_end:]], axis=0)
        y_fold_train = np.concatenate([y_train[:val_start], y_train[val_end:]], axis=0)

        # Reset model for each fold
        adapter = adapter_cls(**adapter_params).to(device)
        adapter = adapters.MultichannelProjector(
            num_channels=n_features,
            new_num_channels=n_components,
            patch_window_size=None,
            base_projector=adapter,
            device=device,
        )
        adapts_model = ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=n_features,
            n_components=n_components,
        )

        # Train on this fold
        adapts_model.fine_tune_iclearner(
            X=X_train,
            y=y_train,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            verbose=0,
            use_adapter=False,
            n_epochs=50,
            seed=seed,
        )
        adapts_model.adapter_supervised_fine_tuning(
            X_fold_train,
            y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            **training_params,
        )

        # Evaluate on validation fold
        with torch.no_grad():
            fold_time_series = np.concatenate([X_fold_val, y_fold_val], axis=-1)
            _, _, _, _ = adapts_model.predict_multi_step(
                X=fold_time_series,
                prediction_horizon=forecasting_horizon,
            )
            fold_metrics.append(adapts_model.compute_metrics())

    # Calculate average metrics across folds
    metrics = {}
    for key in fold_metrics[0].keys():
        metrics[key] = np.mean([fm[key] for fm in fold_metrics])

    # Final evaluation on test set using the last model
    with torch.no_grad():
        _, _, _, _ = adapts_model.predict_multi_step(
            X=time_series_test,
            prediction_horizon=forecasting_horizon,
        )
        test_metrics = adapts_model.compute_metrics(
            logdir=Path(train.get_context().get_trial_dir())
        )

    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Save metrics to CSV
    config.update({"train_size": train_size})
    config.update({"training": "full"})
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
    k_fold: int = 3,
    metric: str = "mse",
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
            k_fold,
        )

    # Set up tuner with scheduler and resources per trial
    trainable_with_gpu = tune.with_resources(
        objective, {"cpu": 1, "gpu": gpu_fraction_per_worker}
    )
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric=metric,  # Change this to the metric you care about
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
    if args.adapter not in NOT_FULL_COMP_ADAPTERS:
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
            k_fold=args.k_fold,
            metric=args.metric,
        )
        print(f"Best config for {args.adapter} with n_comp {n_components}:")
        print(best_config)
