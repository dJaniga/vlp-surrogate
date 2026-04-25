"""
Runner on top of `main.py` that executes the pipeline for every supported
model type with hyperparameter optimization enabled.

Usage example:
    python run_all_models.py \
        --input-file path/to/sim.UNSMRY \
        --vfp-details-file path/to/vfp.json \
        --output-folder path/to/out \
        [--tuning-metric mean_squared_error] \
        [--seed 42] \
        [--models linear xgb gp]   # optional subset
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

from toolbox import setup_logging
from vfp import run_pipeline
from vfp.api import create_model
from vfp.modeling.tuning_metrics import AVAILABLE_METRICS

logger = logging.getLogger(__name__)

ALL_MODELS: tuple[str, ...] = ("linear", "elasticnet", "bayesian_ridge", "symbolic")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VFP pipeline for all model types with hyperparameter optimization.",
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
        help="Root output folder. Each model writes to a subfolder.",
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
    parser.add_argument(
        "--tuning-metric",
        type=str,
        choices=AVAILABLE_METRICS,
        default="mean_squared_error",
        help="Metric to optimize during hyperparameter tuning.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
        help="Subset of models to run (default: all).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue with the next model if one fails (default: True).",
    )
    return parser


def _build_model(model_type: str, seed: int | None):
    """Construct a model with default constructor args; tuning will explore the space."""
    # Hyperparameter optimization will override architectural choices,
    # so default constructors are sufficient here.
    return create_model(model_type, seed=seed)


def run_for_model(model_type: str, args: argparse.Namespace) -> None:
    logger.info("=" * 72)
    logger.info("Starting model: %s", model_type)
    logger.info("=" * 72)

    model_output = args.output_folder / model_type
    model_output.mkdir(parents=True, exist_ok=True)

    model = _build_model(model_type, args.seed)

    run_pipeline(
        source_file_path=args.input_file,
        vfp_details_file_path=args.vfp_details_file,
        surrogate_model=model,
        output_folder_path=model_output,
        well_data_filter_path=args.well_data_filter_file,
        table_granularity=args.table_granularity,
        optimize_hyperparameters=True,
        tuning_metric=args.tuning_metric,
    )


def main() -> int:
    setup_logging()
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.output_folder.exists():
        logger.warning("Cleaning output folder: %s", args.output_folder)
        shutil.rmtree(args.output_folder)

    args.output_folder.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    for model_type in args.models:
        try:
            run_for_model(model_type, args)
            results[model_type] = "OK"
        except Exception as exc:  # noqa: BLE001
            results[model_type] = f"FAILED: {exc.__class__.__name__}: {exc}"
            logger.error("Model %s failed:\n%s", model_type, traceback.format_exc())
            if not args.continue_on_error:
                break

    logger.info("=" * 72)
    logger.info("Run summary:")
    for model_type, status in results.items():
        logger.info("  %-15s %s", model_type, status)
    logger.info("=" * 72)

    return 0 if all(v == "OK" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
