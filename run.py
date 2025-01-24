import os
import json
from pathlib import Path
import time
import random

from dataclasses import dataclass
from typing import Optional
import tyro

import numpy as np
import torch

# DICL
from dicl import dicl, adapters
from dicl.icl import iclearner as icl
from dicl.utils.main_script import (
    save_metrics_to_csv,
    setup_logging,
    prepare_data,
    load_moment_model,
    load_moirai_model,
)
from dicl.utils.preprocessing import get_gpu_memory_stats
from dicl.adapters import (
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
)


os.environ["HF_HOME"] = "/mnt/vdb/hugguingface/"
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
    data_path: Path = Path("/mnt/vdb/abenechehab/dicl-adapters/results/data.csv")
    seed: int = 13
    device: str = "cpu"
    logger_name: str = "DICL Adapter"
    log_level: str = "INFO"
    log_dir: Path = Path("/mnt/vdb/abenechehab/dicl-adapters/logs/logger")
    number_n_comp_to_try: int = 4
    inference_batch_size: int = 512
    supervised: bool = False
    use_revin: bool = False
    pca_in_preprocessing: bool = False
    custom_n_comp: bool = False


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
            if args.adapter in ["simpleAE", "VAE"]:
                adapter_params.update(
                    {
                        "num_layers": adapter_config["num_layers"],
                        "hidden_dim": adapter_config["hidden_dim"],
                    }
                )
            elif args.adapter in ["flow", "AEflow"]:
                adapter_params.update(
                    {
                        "num_coupling": adapter_config["num_coupling"],
                        "hidden_dim": adapter_config["hidden_dim"],
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

        disentangler = adapters.MultichannelProjector(
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

        DICL = dicl.DICL(
            disentangler=disentangler,
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
        if args.supervised:
            # assert not args.is_fine_tuned, "iclearner must be frozen when adapter is "
            # "(supervised) fine-tuned"
            DICL.adapter_supervised_fine_tuning(
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
        else:
            DICL.fit_disentangler(X=np.concatenate([X_train, X_val], axis=0))
        if args.adapter and args.adapter not in ["pca"]:
            torch.save(
                DICL.disentangler.base_projector_,
                Path(log_dir) / f"n_comp_{n_components}/" / "adapter.pt",
            )

        logger.info(
            f"[{n_components}/{start}:{end}] adapter fitted (supervised:"
            f"{args.supervised}) and saved in {time.time() - next_time_cp:.2f} seconds"
        )
        next_time_cp = time.time()

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
            logger.info(
                f"[{n_components}/{start}:{end}] iclearner fine-tuned in "
                f"{time.time() - next_time_cp:.2f} seconds"
            )
            next_time_cp = time.time()

        with torch.no_grad():
            _, _, _, _ = DICL.predict_multi_step(
                X=time_series,
                prediction_horizon=args.forecast_horizon,
                batch_size=args.inference_batch_size,
            )

        logger.info(
            f"[{n_components}/{start}:{end}] multi-step prediction done in "
            f"{time.time() - next_time_cp:.2f} seconds"
        )
        next_time_cp = time.time()

        metrics = DICL.compute_metrics()
        logger.info(
            f"[{n_components}/{start}:{end}] metrics computed in "
            f"{time.time() - next_time_cp:.2f} seconds"
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
            is_fine_tuned="supervised" if args.supervised else args.is_fine_tuned,
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
        del disentangler, iclearner, DICL

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
