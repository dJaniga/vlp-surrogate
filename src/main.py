from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vfp import VFPTablePipeline, create_model
from vfp.utils.logging import setup_logging
from vfp.models import VFPDataConfig, VFPTableConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VFP tables or run GA regression demo."
    )
    parser.add_argument("--file-path", type=Path, help="Input XLSX path.")
    parser.add_argument(
        "--file-type",
        choices=["p", "i"],
        help="File type: p (production) or i (injection).",
    )
    parser.add_argument("--output", type=Path, help="Output VFP table path.")
    parser.add_argument(
        "--model",
        choices=["linear", "ga"],
        default="linear",
        help="Model type for regression.",
    )
    parser.add_argument("--data-start-line", type=int, default=5)
    parser.add_argument("--min-flow", type=float, default=500.0)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--table-number", type=int, default=1)
    parser.add_argument("--well-depth", type=float, default=0.0)
    parser.add_argument("--ga-generations", type=int, default=80)
    parser.add_argument("--ga-population", type=int, default=60)
    parser.add_argument("--ga-seed", type=int, default=None)
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if not args.file_path or not args.file_type or not args.output:
        parser.error("--file-path, --file-type, and --output are required.")

    logger.info("Building VFP configs", extra={"file_path": str(args.file_path)})
    data_config = VFPDataConfig(
        file_path=args.file_path,
        file_type=args.file_type,
        data_start_line=args.data_start_line,
        minimum_acceptable_flow=args.min_flow,
        start_time=args.start_time,
    )
    table_config = VFPTableConfig(
        vfp_parameter_size=args.grid_size,
        vlp_table_number=args.table_number,
        well_depth=args.well_depth,
    )

    if args.model == "ga":
        logger.info(
            "Using GA model",
            extra={
                "generations": args.ga_generations,
                "population": args.ga_population,
            },
        )
        model = create_model(
            "ga",
            generations=args.ga_generations,
            population_size=args.ga_population,
            seed=args.ga_seed,
        )
    else:
        logger.info("Using linear model", extra={"model": "linear"})
        model = create_model("linear")

    pipeline = VFPTablePipeline(data_config, table_config, model=model)
    pipeline.run(args.output)
    logger.info("VFP table saved", extra={"output": str(args.output)})


if __name__ == "__main__":
    main()
