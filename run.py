import os
import json
from pathlib import Path
import time
import random
import warnings

from dataclasses import dataclass
from typing import Optional
import tyro

import numpy as np
import torch

# adapts
from adapts import adapts, adapters
from adapts.icl import iclearner as icl
from adapts.utils.main_script import (
    save_metrics_to_csv,
    setup_logging,
    prepare_data,
    load_moment_model,
    load_moirai_model,
)
from adapts.utils.preprocessing import get_gpu_memory_stats
from adapts.adapters import (
    SimpleAutoEncoder,
    LinearAutoEncoder,
    betaVAE,
    NormalizingFlow,
    AENormalizingFlow,
    JustRevIn,
    betaLinearVAE,
    DropoutLinearAutoEncoder,
    LinearDecoder,
    LinearEncoder,
    likelihoodVAE,
    linearLikelihoodVAE,
)


warnings.filterwarnings("ignore", category=FutureWarning)
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
MAX_TRAIN_SIZE = 10000
CUSTOM_N_COMP = [2, 5, 9, 14]


@dataclass
class Args:
    is_fine_tuned: bool = False
    forecast_horizon: int = 96
    model_name: str = "AutonLab/MOMENT-1-large"  # f"Salesforce/moirai-1.1-R-large"
    context_length: int = 512
    dataset_name: str = "ETTh1"  # Will be set based on forecast_horizon
    adapter: Optional[str] = None  # "pca"
    data_path: Path = Path("results/latest.csv")
    seed: int = 13
    device: str = "cpu"
    logger_name: str = "DICL Adapter"
    log_level: str = "INFO"
    log_dir: Path = Path("logs/latest")
    number_n_comp_to_try: int = 4
    inference_batch_size: int = 128
    supervised: str = "False"
    use_revin: bool = False
    pca_in_preprocessing: bool = False
    custom_n_comp: bool = False
    n_epochs_fine_tuning: int = 50
    n_epochs_adapter: int = 100


def main(args: Args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set dataset name based on forecast horizon if not provided
    dataset_name = f"{args.dataset_name}_pred={args.forecast_horizon}"

    logger, log_dir = setup_logging(
        args.logger_name,
        args.log_level,
        args.log_dir,
        dataset_name,
        args.adapter,
        args.model_name.split("/")[-1],
    )

    # Write args as json to config file in log directory
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(Path(log_dir) / "config.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    start_time = time.time()
    logger.info(f"Starting data preparation for dataset: {dataset_name}")

    X_train, y_train, X_val, y_val, X_test, y_test, n_features = prepare_data(
        dataset_name, args.context_length, args.forecast_horizon
    )
    time_series = np.concatenate([X_test, y_test], axis=-1)
    # Limit training size
    if len(X_train) > MAX_TRAIN_SIZE:
        indices = np.random.choice(len(X_train), MAX_TRAIN_SIZE, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_size = MAX_TRAIN_SIZE
    else:
        train_size = len(X_train)

    logger.info(
        f"Test Data shape: {time_series.shape}. X_train shape: {X_train.shape}."
        f"Preparation completed in {time.time() - start_time:.2f} seconds"
    )

    start = 1 if args.adapter in NOT_FULL_COMP_ADAPTERS else n_features
    end = n_features
    number_n_comp_to_try = 1 if not args.adapter else args.number_n_comp_to_try

    model_loaded = False

    if args.custom_n_comp:
        possible_n_components = CUSTOM_N_COMP
    else:
        if end > start:
            possible_n_components = np.linspace(
                start, end, min(number_n_comp_to_try, end - start)
            ).astype("int32")
        else:
            possible_n_components = [n_features]

    logger.info(
        f"n_components to try between {possible_n_components[0]} and "
        f"{possible_n_components[-1]}: {possible_n_components}"
    )

    for n_components in possible_n_components:
        start_time = time.time()

        if (not model_loaded) or args.is_fine_tuned:
            model_loaded = True
            logger.info(
                f"[{n_components}/{start}:{end}] Starting loading model: "
                f"{args.model_name}"
            )
            if "MOMENT" in args.model_name:
                model = load_moment_model(args.model_name, args.forecast_horizon).to(
                    torch.device(args.device)
                )
                icl_constructor = icl.MomentICLTrainer
            elif "moirai" in args.model_name:
                model = load_moirai_model(
                    args.model_name, args.forecast_horizon, args.context_length
                ).to(torch.device(args.device))
                icl_constructor = icl.MoiraiICLTrainer
            elif "ttm" in args.model_name:
                raise NotImplementedError("TTM backbone not implemented yet")
            else:
                raise ValueError(f"Not supported model: {args.model_name}")
            logger.info(
                f"[{n_components}/{start}:{end}] Model loaded in "
                f"{time.time() - start_time:.2f} seconds"
            )

        config_file_path = Path(
            f"results/config/{args.dataset_name}_{args.adapter}.json"
        )
        if args.adapter and (args.adapter in ADAPTER_CLS) and config_file_path.exists():
            with open(config_file_path, "r") as f:
                adapter_config = json.load(f)

            # Configure adapter
            adapter_params = {
                "input_dim": n_features,
                "device": args.device,  # Use the determined device
                "context_length": args.context_length,
                "forecast_horizon": args.forecast_horizon,
                "use_revin": args.use_revin,  # use_revin might be in config as well
            }
            if args.adapter != "flow":
                adapter_params.update(
                    {
                        "n_components": n_components,
                    }
                )
            if args.adapter in ["simpleAE", "VAE", "lVAE"]:
                adapter_params.update(
                    {
                        "num_layers": adapter_config["num_layers"],
                        "hidden_dim": adapter_config["hidden_dim"],
                    }
                )
            if args.adapter in ["flow", "AEflow"]:
                adapter_params.update(
                    {
                        "num_coupling": adapter_config["num_coupling"],
                        "hidden_dim": adapter_config["hidden_dim"],
                    }
                )
            if "VAE" in args.adapter:
                adapter_params.update(
                    {
                        "beta": adapter_config["beta"],
                    }
                )
            if "dropout" in args.adapter or args.adapter == "AEflow":
                adapter_params.update(
                    {
                        "p": adapter_config["p"],
                    }
                )

            adapter = ADAPTER_CLS[args.adapter](**adapter_params).to(args.device)
            learning_rate = adapter_config["learning_rate"]
            batch_size = adapter_config["batch_size"]
        else:
            if not args.adapter and args.use_revin:
                adapter = JustRevIn(
                    num_features=n_features,
                    context_length=args.context_length,
                    forecast_horizon=args.forecast_horizon,
                    device=args.device,
                ).to(args.device)
            else:
                adapter = args.adapter
            learning_rate = 0.001
            batch_size = 32

        adapter = adapters.MultichannelProjector(
            num_channels=n_features,
            new_num_channels=n_components,
            patch_window_size=None,
            base_projector=adapter,
            device=args.device,
            use_revin=args.use_revin,
            context_length=args.context_length,
            forecast_horizon=args.forecast_horizon,
        )

        iclearner = icl_constructor(
            model=model,
            n_features=n_components,
            forecast_horizon=args.forecast_horizon,
        )

        adapts_model = adapts.ADAPTS(
            adapter=adapter,
            iclearner=iclearner,
            n_features=n_features,
            n_components=n_components,
            pca_in_preprocessing=args.pca_in_preprocessing,
        )

        logger.info(
            f"[{n_components}/{start}:{end}] Starting fitting adapter: "
            f"{args.adapter}"
        )
        next_time_cp = time.time()
        os.makedirs(Path(log_dir) / f"n_comp_{n_components}", exist_ok=True)
        if args.supervised == "True":
            adapts_model.adapter_supervised_fine_tuning(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                device=args.device,
                log_dir=Path(log_dir) / f"n_comp_{n_components}",
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=1,
                n_epochs=args.n_epochs_adapter,
                logger=logger,
            )
        elif args.supervised == "full":
            adapts_model.adapter_and_head_supervised_fine_tuning(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                device=args.device,
                log_dir=Path(log_dir) / f"n_comp_{n_components}",
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=1,
            )
        elif args.supervised == "ft_then_supervised":
            adapts_model.fine_tune_iclearner(
                X=X_train,
                y=y_train,
                batch_size=batch_size,
                learning_rate=learning_rate,
                verbose=1,
                use_adapter=False,
                n_epochs=args.n_epochs_fine_tuning,
                logger=logger,
                seed=args.seed,
            )
            logger.info(
                f"[{n_components}/{start}:{end}] Done fine tuning, now training adapter"
            )
            adapts_model.adapter_supervised_fine_tuning(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                device=args.device,
                log_dir=Path(log_dir) / f"n_comp_{n_components}",
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=1,
                n_epochs=args.n_epochs_adapter,
                logger=logger,
            )
        elif args.supervised == "bilevel":
            adapts_model.fine_tune_iclearner(
                X=X_train,
                y=y_train,
                batch_size=batch_size,
                learning_rate=learning_rate,
                verbose=1,
                use_adapter=False,
                n_epochs=args.n_epochs_fine_tuning,
                logger=logger,
            )
            logger.info(
                f"[{n_components}/{start}:{end}] Done fine tuning, now training adapter"
            )
            adapts_model.adapter_supervised_fine_tuning(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                device=args.device,
                log_dir=Path(log_dir) / f"n_comp_{n_components}",
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=1,
                n_epochs=100,
                logger=logger,
            )
            logger.info(
                f"[{n_components}/{start}:{end}] Done training adapter, now fine "
                "tuning again ;)"
            )
            adapts_model.fine_tune_iclearner(
                X=X_train,
                y=y_train,
                batch_size=batch_size,
                learning_rate=learning_rate,
                verbose=1,
                use_adapter=True,
                n_epochs=50,
                logger=logger,
            )
        elif args.supervised == "ft":
            adapts_model.fine_tune_iclearner(
                X=X_train,
                y=y_train,
                batch_size=batch_size,
                learning_rate=learning_rate,
                verbose=1,
                use_adapter=False,
                n_epochs=args.n_epochs_fine_tuning,
                logger=logger,
            )
            adapts_model.fit_adapter(X=np.concatenate([X_train, X_val], axis=0))
        elif args.supervised == "unsupervised":
            adapts_model.fine_tune_iclearner(
                X=X_train,
                y=y_train,
                batch_size=batch_size,
                learning_rate=learning_rate,
                verbose=1,
                use_adapter=False,
                n_epochs=50,
                logger=logger,
                seed=args.seed,
            )
            logger.info(
                f"[{n_components}/{start}:{end}] Done fine tuning, now training adapter"
            )
            adapts_model.fit_adapter(X=np.concatenate([X_train, X_val], axis=0))
        elif args.supervised == "False":
            adapts_model.fit_adapter(X=np.concatenate([X_train, X_val], axis=0))
        else:
            raise ValueError(f"Invalid supervised argument: {args.supervised}")
        if args.adapter and args.adapter not in ["pca"]:
            torch.save(
                adapts_model.adapter.base_projector_,
                Path(log_dir) / f"n_comp_{n_components}/" / "adapter.pt",
            )

        logger.info(
            f"[{n_components}/{start}:{end}] adapter fitted (supervised:"
            f"{args.supervised}) and saved in {time.time() - next_time_cp:.2f} seconds"
        )
        next_time_cp = time.time()

        with torch.no_grad():
            _, _, _, _ = adapts_model.predict_multi_step(
                X=time_series,
                prediction_horizon=args.forecast_horizon,
                batch_size=args.inference_batch_size,
                n_samples=25,
            )

        logger.info(
            f"[{n_components}/{start}:{end}] multi-step prediction done in "
            f"{time.time() - next_time_cp:.2f} seconds"
        )
        next_time_cp = time.time()

        metrics = adapts_model.compute_metrics(
            logdir=Path(log_dir) / f"n_comp_{n_components}"
        )
        logger.info(
            f"[{n_components}/{start}:{end}] metrics [mse={metrics['mse']},"
            f"mae={metrics['mae']}] computed in {time.time() - next_time_cp:.2f} sec"
        )

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
            is_fine_tuned=args.supervised,
            pca_in_preprocessing=args.pca_in_preprocessing,
            use_revin=args.use_revin,
            elapsed_time=time.time() - start_time,
            seed=args.seed,
            train_size=train_size,
        )

        logger.info(
            f"[{n_components}/{start}:{end}] overall runtime "
            f"{time.time() - start_time:.2f} seconds"
        )

        # clean memory
        del adapter, iclearner, adapts_model

        gpu_stats = get_gpu_memory_stats()
        for gpu_id in gpu_stats:
            if "allocated" in gpu_id:
                gpu_num = gpu_id.split("_")[1]
                allocated = gpu_stats[f"gpu_{gpu_num}_allocated(%)"]
                reserved = gpu_stats[f"gpu_{gpu_num}_reserved(%)"]
                logger.info(
                    f"GPU {gpu_num} - Reserved: {reserved:.2f}%, Allocated: "
                    f"{allocated:.2f}%"
                )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
