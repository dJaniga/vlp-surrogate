"""Collect fit-results JSON files produced for VFP tables into tidy DataFrames.

Folder layout expected:
    vfp_tables/
        <method>/
            <optimization_metric>/
                <WELL>_p/   # production well
                    *_fit_results.json
                <WELL>_i/   # injection well
                    *_fit_results.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

WELL_KIND = {"p": "production", "i": "injection"}


def _iter_result_files(root: Path) -> Iterator[tuple[Path, str, str, str, str]]:
    """Yield (json_path, method, opt_metric, well_name, well_kind)."""
    for json_path in root.glob("*/*/*_?/*_fit_results.json"):
        well_dir = json_path.parent
        suffix = well_dir.name.rsplit("_", 1)[-1].lower()
        if suffix not in WELL_KIND:
            continue
        well_name = well_dir.name[: -(len(suffix) + 1)]
        opt_metric = well_dir.parent.name
        method = well_dir.parent.parent.name
        yield json_path, method, opt_metric, well_name, WELL_KIND[suffix]


def collect(root: Path) -> pd.DataFrame:
    """Walk *root* and return a long-format DataFrame of every metric value."""
    rows: list[dict] = []
    for json_path, method, opt_metric, well, kind in _iter_result_files(root):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            print(f"[skip] {json_path}: {err}")
            continue

        for split, metrics in payload.items():  # train_resubstitution / nested_cv
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "method": method,
                        "optimization_metric": opt_metric,
                        "well": well,
                        "well_kind": kind,
                        "split": split,
                        "metric": metric_name,
                        "value": value,
                        "source": str(json_path.relative_to(root)),
                    }
                )

    return pd.DataFrame(rows)


def export(df: pd.DataFrame, out_dir: Path) -> None:
    """Save one combined CSV plus per-method/metric/split wide tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_results_long.csv", index=False)

    # Wide pivot: rows = wells, cols = metric, separated per kind/method/opt_metric/split
    for (kind, method, opt_metric, split), chunk in df.groupby(
        ["well_kind", "method", "optimization_metric", "split"], dropna=False
    ):
        wide = chunk.pivot_table(
            index="well", columns="metric", values="value", aggfunc="first"
        ).sort_index()
        sub = out_dir / kind / method / opt_metric
        sub.mkdir(parents=True, exist_ok=True)
        wide.to_csv(sub / f"{split}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect VFP fit results.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("vfp_tables"),
        help="Path to the vfp_tables folder (default: ./vfp_tables).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("vfp_tables_summary"),
        help="Output directory for aggregated CSVs.",
    )
    args = parser.parse_args()

    df = collect(args.root)
    if df.empty:
        print(f"No fit-results JSON files found under {args.root.resolve()}")
        return

    export(df, args.out)
    print(
        f"Collected {len(df):,} metric rows from "
        f"{df['source'].nunique()} files → {args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
