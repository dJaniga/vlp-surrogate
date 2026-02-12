from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ____vfp import VFPTablePipeline, create_model
from ____vfp.utils.logging import setup_logging
from ____vfp.models import VFPDataConfig, VFPTableConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate VFP tables.")
    parser.add_argument("--file-path", type=Path, help="Input XLSX path.")
    parser.add_argument(
        "--file-type",
        choices=["p", "i"],
        help="File type: p (production) or i (injection).",
    )
    parser.add_argument("--output", type=Path, help="Output VFP table path.")
    parser.add_argument(
        "--model",
        choices=["linear", "ga", "symbolic"],
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

    # Symbolic regressor specific arguments
    parser.add_argument(
        "--n-islands",
        type=int,
        default=4,
        help="Number of islands for symbolic island-model GP.",
    )
    parser.add_argument(
        "--migration-interval",
        type=int,
        default=5,
        help="Generations between island migrations.",
    )
    parser.add_argument(
        "--migration-size",
        type=int,
        default=3,
        help="Number of individuals migrated between islands.",
    )
    parser.add_argument(
        "--simplify-interval",
        type=int,
        default=5,
        help="Generations between SymPy simplification passes (0 to disable).",
    )
    parser.add_argument(
        "--parsimony-coefficient",
        type=float,
        default=0.001,
        help="Parsimony pressure coefficient (penalty per tree node).",
    )
    parser.add_argument(
        "--max-tree-height",
        type=int,
        default=6,
        help="Maximum GP tree height.",
    )
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
    elif args.model == "symbolic":
        logger.info("Using symbolic model", extra={"model": "symbolic"})
        model = create_model(
            "symbolic",
            generations=args.ga_generations,
            population_size=args.ga_population,
            seed=args.ga_seed,
            n_islands=args.n_islands,
            migration_interval=args.migration_interval,
            migration_size=args.migration_size,
            simplify_interval=args.simplify_interval,
            parsimony_coefficient=args.parsimony_coefficient,
            max_tree_height=args.max_tree_height,
        )
    else:
        logger.info("Using linear model", extra={"model": "linear"})
        model = create_model("linear")

    pipeline = VFPTablePipeline(data_config, table_config, model=model)
    pipeline.run(args.output)
    logger.info("VFP table saved", extra={"output": str(args.output)})


if __name__ == "__main__":
    main()
