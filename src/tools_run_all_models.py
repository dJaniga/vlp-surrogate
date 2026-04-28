from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from toolbox import setup_logging
from vfp import run_pipeline
from vfp.api import create_model
from vfp.modeling.tuning_metrics import AVAILABLE_METRICS

logger = logging.getLogger(__name__)

ALL_MODELS: tuple[str, ...] = ("linear", "elasticnet", "bayesian_ridge", "huber", "symbolic")

SELECTED_METRICS: tuple[str, ...] = (
    "mean_absolute_percentage_error",
    "root_mean_squared_log_error",
    "median_absolute_error",
    "r2_score",
)


def _detect_physical_cpu_count() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except ImportError:
        pass

    logical = os.cpu_count() or 1
    return max(1, logical // 2) if logical > 1 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VFP pipeline for every (model, metric) combination "
        "with hyperparameter optimization, in parallel.",
        prog="vfp-surrogate-runner",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to simulation results file [*.UNSMRY]",
    )
    parser.add_argument(
        "--vfp-details-file",
        type=Path,
        required=True,
        help="Path to VFP details file [*.json]",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Root output folder. Each (model, metric) pair "
        "writes to <output>/<model>/<metric>/.",
    )
    parser.add_argument(
        "--well-data-filter-file",
        type=Path,
        default=None,
        help="Path to well data filter file [*.json]",
    )
    parser.add_argument(
        "--table-granularity", type=int, default=5, help="VFP records n-size"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel processes (default: physical CPU core count).",
    )
    return parser


def _build_model(model_type: str, seed: int | None):
    """Construct a model with default constructor args; tuning will explore the space."""
    return create_model(model_type, seed=seed)


def _run_combination_worker(
    model_type: str,
    metric_name: str,
    input_file: Path,
    vfp_details_file: Path,
    output_folder: Path,
    well_data_filter_file: Path | None,
    table_granularity: int,
    seed: int | None,
) -> tuple[str, str, str]:
    """
    Worker entrypoint executed in a child process.

    Returns (model_type, metric_name, status). Never raises — failures are
    captured into the status string so the parent can summarize them.
    """
    # Each worker needs its own logging configuration.
    setup_logging()
    worker_logger = logging.getLogger(__name__)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    combo_output = output_folder / model_type / metric_name
    combo_output.mkdir(parents=True, exist_ok=True)

    worker_logger.info(
        "[pid=%d] Starting: model=%s | metric=%s", os.getpid(), model_type, metric_name
    )
    try:
        model = _build_model(model_type, seed)
        run_pipeline(
            source_file_path=input_file,
            vfp_details_file_path=vfp_details_file,
            surrogate_model=model,
            output_folder_path=combo_output,
            well_data_filter_path=well_data_filter_file,
            table_granularity=table_granularity,
            optimize_hyperparameters=True,
            tuning_metric=metric_name,
        )
        return model_type, metric_name, "OK"
    except Exception as exc:  # noqa: BLE001
        worker_logger.error(
            "[pid=%d] Combination model=%s metric=%s failed:\n%s",
            os.getpid(),
            model_type,
            metric_name,
            traceback.format_exc(),
        )
        return model_type, metric_name, f"FAILED: {exc.__class__.__name__}: {exc}"


def main() -> int:
    setup_logging()
    args = build_parser().parse_args()

    _unknown_metrics = set(SELECTED_METRICS) - set(AVAILABLE_METRICS)
    if _unknown_metrics:
        raise ValueError(
            f"SELECTED_METRICS contains unknown metric(s): {sorted(_unknown_metrics)}. "
            f"Allowed values: {AVAILABLE_METRICS}"
        )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.output_folder.exists():
        logger.warning("Cleaning output folder: %s", args.output_folder)
        shutil.rmtree(args.output_folder)
    args.output_folder.mkdir(parents=True)

    combinations: list[tuple[str, str]] = [
        (m, metric) for m in ALL_MODELS for metric in SELECTED_METRICS
    ]
    total = len(combinations)

    physical_cpus = _detect_physical_cpu_count()
    max_workers = args.max_workers or physical_cpus
    max_workers = max(1, min(max_workers, total))

    logger.info(
        "Running %d combination(s): %d models × %d metrics | workers=%d (physical CPUs=%d)",
        total,
        len(ALL_MODELS),
        len(SELECTED_METRICS),
        max_workers,
        physical_cpus,
    )

    results: dict[tuple[str, str], str] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, (model_type, metric_name) in enumerate(combinations):
            task_seed = None if args.seed is None else args.seed + idx
            future = executor.submit(
                _run_combination_worker,
                model_type,
                metric_name,
                args.input_file,
                args.vfp_details_file,
                args.output_folder,
                args.well_data_filter_file,
                args.table_granularity,
                task_seed,
            )
            futures[future] = (model_type, metric_name)

        completed = 0
        for future in as_completed(futures):
            model_type, metric_name, status = future.result()
            results[(model_type, metric_name)] = status
            completed += 1
            logger.debug(
                "[%d/%d] %-15s | %-32s -> %s",
                completed,
                total,
                model_type,
                metric_name,
                status,
            )

    logger.info("=" * 72)
    logger.info("Run summary (%d combinations):", len(results))
    for (model_type, metric_name), status in sorted(results.items()):
        logger.info("  %-15s | %-32s %s", model_type, metric_name, status)
    ok_count = sum(1 for v in results.values() if v == "OK")
    logger.info("Succeeded: %d / %d", ok_count, len(results))
    logger.info("=" * 72)

    return 0 if ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
