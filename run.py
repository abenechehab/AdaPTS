import os
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
    logger_name: str = "DICL Adapter"
    log_level: str = "INFO"
    log_dir: Path = Path("/mnt/vdb/abenechehab/dicl-adapters/logs/logger")
    number_n_comp_to_try: int = 4
    inference_batch_size: int = 512
    supervised: bool = False


def main(args: Args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger, log_dir = setup_logging(args.logger_name, args.log_level, args.log_dir)

    # Set dataset name based on forecast horizon if not provided
    dataset_name = f"{args.dataset_name}_pred={args.forecast_horizon}"

    start_time = time.time()
    logger.info(f"Starting data preparation for dataset: {dataset_name}")

    X_train, y_train, X_val, y_val, X_test, y_test, n_features = prepare_data(
        dataset_name, args.context_length, args.forecast_horizon
    )
    time_series = np.concatenate([X_test, y_test], axis=-1)

    logger.info(
        f"Data shape: {time_series.shape}. "
        f"Preparation completed in {time.time() - start_time:.2f} seconds"
    )

    start = n_features if not args.adapter else 1
    end = n_features
    number_n_comp_to_try = 1 if not args.adapter else args.number_n_comp_to_try

    model_loaded = False

    if end > start:
        possible_n_components = np.linspace(
            start, end, min(number_n_comp_to_try, end - start)
        ).astype("int32")
    else:
        possible_n_components = [n_features]

    logger.info(
        f"n_components to try between {start} and {end}: " f"{possible_n_components}"
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

        disentangler = adapters.MultichannelProjector(
            num_channels=n_features,
            new_num_channels=n_components,
            patch_window_size=None,
            base_projector=args.adapter,
            device=args.device,
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
        )

        logger.info(
            f"[{n_components}/{start}:{end}] Starting fitting adapter: "
            f"{args.model_name}"
        )
        next_time_cp = time.time()
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
            )
        else:
            DICL.fit_disentangler(X=np.concatenate([X_train, X_val], axis=0))

        logger.info(
            f"[{n_components}/{start}:{end}] adapter fitted in "
            f"{time.time() - next_time_cp:.2f} seconds"
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
            elapsed_time=time.time() - start_time,
            seed=args.seed,
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
