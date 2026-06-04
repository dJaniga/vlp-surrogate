import argparse
import logging
import os
from pathlib import Path
import random

import numpy as np

from toolbox import setup_logging
from vfp import run_pipeline, run_evaluator
from vfp.api import create_model
from vfp.modeling.tuning_metrics import AVAILABLE_METRICS

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VFP table generator & verification.", prog="vfp-surrogate"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_pipeline = subparsers.add_parser(
        "pipeline",
        help="Pipeline mode - build VFP surrogate model from simulation results and export it to VFP format.",
    )
    parser_pipeline.add_argument(
        "--input-file",
        type=Path,
        help="Path to simulation results file [*.UNSMRY]",
        required=True,
    )
    parser_pipeline.add_argument(
        "--vfp-details-file",
        type=Path,
        help="Path to VFP details file [*.json]",
        required=True,
    )
    parser_pipeline.add_argument(
        "--model",
        type=str,
        help="VFP surrogate model type",
        required=True,
        choices=[
            "linear",
            "symbolic",
            "xgb",
            "gp",
            "elasticnet",
            "bayesian_ridge",
            "huber",
        ],
        default="linear",
    )

    parser_pipeline.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser_pipeline.add_argument(
        "--output-folder", type=Path, help="Path to output folder", required=True
    )
    parser_pipeline.add_argument(
        "--well-data-filter-file",
        type=Path,
        help="Path to well data filter file [*.json]",
    )
    parser_pipeline.add_argument(
        "--table-granularity", type=int, default=5, help="VFP records n-size"
    )

    parser_pipeline.add_argument(
        "--optimize-hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning (e.g. Optuna)",
        default=False,
    )
    parser_pipeline.add_argument(
        "--tuning-metric",
        type=str,
        help="Metric to optimize during hyperparameter tuning (e.g., mean_squared_error, r2_score).",
        choices=AVAILABLE_METRICS,
        default="mean_squared_error",
    )

    parser_evaluator = subparsers.add_parser(
        "evaluator",
        help="Evaluator mode - evaluate VFP surrogate model against simulation results.",
    )
    parser_evaluator.add_argument(
        "--input-file",
        type=Path,
        help="Path to simulation results file [*.UNSMRY]",
        required=True,
    )
    parser_evaluator.add_argument(
        "--output-folder", type=Path, help="Path to output folder", required=True
    )
    parser_evaluator.add_argument(
        "--well-data-filter-file",
        type=Path,
        help="Path to well data filter file [*.json]",
    )

    parser_evaluator.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser


def main():

    parser = build_parser()
    parsed_args = parser.parse_args()

    if parsed_args.mode == "pipeline":
        log_path = (
            parsed_args.output_folder / parsed_args.model / f"{parsed_args.model}.log"
        )
    else:
        log_path = None

    setup_logging(log_path=log_path)

    os.environ["VLP_FIT_METRIC"] = parsed_args.tuning_metric

    if parsed_args.seed is not None:
        random.seed(parsed_args.seed)
        np.random.seed(parsed_args.seed)

    if parsed_args.mode == "pipeline":
        logger.info("Running in pipeline mode", extra={"Parsed args": parsed_args})

        if parsed_args.model == "symbolic":
            logger.info("Using symbolic model", extra={"model": "symbolic"})
            model = create_model("symbolic", seed=parsed_args.seed)
        elif parsed_args.model == "linear":
            logger.info("Using linear model", extra={"model": "linear"})
            model = create_model("linear", seed=parsed_args.seed)
        elif parsed_args.model == "elasticnet":
            logger.info("Using elasticnet model")
            model = create_model("elasticnet", seed=parsed_args.seed)
        elif parsed_args.model == "xgb":
            logger.info("Using xgb model")
            model = create_model("xgb", seed=parsed_args.seed)
        elif parsed_args.model == "gp":
            logger.info("Using gp model")
            model = create_model("gp", seed=parsed_args.seed)
        elif parsed_args.model == "bayesian_ridge":
            logger.info("Using bayesian ridge model")
            model = create_model("bayesian_ridge", seed=parsed_args.seed)
        elif parsed_args.model == "huber":
            model = create_model("huber", seed=parsed_args.seed)
        else:
            raise ValueError(f"Unsupported model: {parsed_args.model}")

        output_folder = parsed_args.output_folder / parsed_args.model

        run_pipeline(
            source_file_path=parsed_args.input_file,
            vfp_details_file_path=parsed_args.vfp_details_file,
            surrogate_model=model,
            output_folder_path=output_folder,
            well_data_filter_path=parsed_args.well_data_filter_file,
            table_granularity=parsed_args.table_granularity,
            optimize_hyperparameters=parsed_args.optimize_hyperparameters,
            tuning_metric=parsed_args.tuning_metric,
        )
        logger.info("Pipeline completed")

    if parsed_args.mode == "evaluator":
        logger.info("Running in evaluator mode", extra={"Parsed args": parsed_args})
        run_evaluator(
            source_file_path=parsed_args.input_file,
            output_folder_path=parsed_args.output_folder,
            well_data_filter_path=parsed_args.well_data_filter_file,
        )
        logger.info("Evaluator completed")


if __name__ == "__main__":
    main()
