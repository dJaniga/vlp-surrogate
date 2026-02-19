from pathlib import Path
from typing import Sequence

import numpy as np

from vfp.builder import VFPType, VFPTable


def export_VFP_table(
    vfp_table: VFPTable, output_file_path: Path, table_type: VFPType
) -> None:
    records = vfp_table.config_records

    dimension_columns = ["FLO", "THP"]
    if table_type is VFPType.PRODUCTION:
        dimension_columns += ["WFR", "GFR", "ALQ"]

    with output_file_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{table_type}\n")
        f.write(f"{vfp_table.header}\n")

        for col in dimension_columns:
            _write_row(f, records[col].unique())

        for row in vfp_table.body:
            _write_row(f, row)


def export_VFP_manifest(vfp_file_paths: Sequence[Path], output_file_path: Path) -> None:
    with output_file_path.open("w", encoding="utf-8") as f:
        for filepath in vfp_file_paths:
            f.write("INCLUDE\n")
            f.write(f"'{filepath.name}' /\n")


def _write_row(file_obj, values: np.ndarray, float_fmt: str = "{:.6g}") -> None:
    def format_value(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)

    formatted = " ".join(format_value(v) for v in values)
    file_obj.write(f"{formatted} /\n")
