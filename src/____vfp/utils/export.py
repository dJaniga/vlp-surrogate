from __future__ import annotations

from pathlib import Path

import numpy as np

from ____vfp.models import VFPTableConfig


def save_to_vfp_file(
    vfp_table_content: np.ndarray,
    prediction_data: np.ndarray,
    output_path: str | Path,
    table_config: VFPTableConfig,
) -> None:
    output_file = Path(output_path)
    header_size = prediction_data.shape[1]

    flow_values = np.unique(prediction_data[:, 0])
    thp_values = np.unique(prediction_data[:, 1])

    with output_file.open("w", encoding="utf-8") as file_obj:
        if header_size == 4:
            wgr_values = np.unique(prediction_data[:, 2])
            file_obj.write("VFPPROD  \n")
            file_obj.write(f"{table_config.vlp_table_number} {table_config.well_depth}")
            file_obj.write(
                f" '{table_config.flow_definition}' '{table_config.wfr_definition}'"
                f" '{table_config.gfr_definition}' '{table_config.thp_definition}'"
                f" '{table_config.alq_definition}' '{table_config.unit_system}'"
                f" '{table_config.tabulated_quantity}'/  \n"
            )
        else:
            file_obj.write("VFPINJ  \n")
            file_obj.write(f"{table_config.vlp_table_number} {table_config.well_depth}")
            file_obj.write(
                f" '{table_config.flow_definition}' '{table_config.thp_definition}'"
                f" '{table_config.unit_system}' '{table_config.tabulated_quantity}' / \n"
            )

        _write_row(file_obj, flow_values)
        _write_row(file_obj, thp_values)

        if header_size == 4:
            _write_row(file_obj, wgr_values)
            file_obj.write("0 /\n")
            file_obj.write("0 /\n")

        for row in vfp_table_content:
            _write_row(file_obj, row)


def _write_row(file_obj, values: np.ndarray) -> None:
    formatted = " ".join(str(value) for value in values)
    file_obj.write(f"{formatted} / \n")
