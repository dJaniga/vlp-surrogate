import argparse
import logging
from pathlib import Path

from toolbox import setup_logging
from vfp import run_pipeline

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
        "--model-name",
        type=str,
        help="VFP surrogate model type",
        required=True,
        choices=["linear"],
    )
    parser_pipeline.add_argument(
        "--output-folder", type=Path, help="Path to output folder", required=True
    )
    parser_pipeline.add_argument(
        "--well-data-filter-file",
        type=Path,
        help="Path to well data filter file [*.json]",
    )
    parser_pipeline.add_argument("--table-granularity", type=int, default=5)

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
        "--well-data-filter-file",
        type=Path,
        help="Path to well data filter file [*.json]",
    )

    return parser


def main():
    setup_logging()
    parser = build_parser()
    parsed_args = parser.parse_args()

    if parsed_args.mode == "pipeline":
        logger.info("Running in pipeline mode", extra={"Parsed args": parsed_args})
        run_pipeline(
            source_file_path=parsed_args.input_file,
            vfp_details_file_path=parsed_args.vfp_details_file,
            model_name=parsed_args.model_name,
            output_folder_path=parsed_args.output_folder,
            well_data_filer_path=parsed_args.well_data_filter_file,
            table_granularity=parsed_args.table_granularity,
        )

    if parsed_args.mode == "evaluator":
        logger.info("Running in evaluator mode", extra={"Parsed args": parsed_args})
        pass


if __name__ == "__main__":
    main()
