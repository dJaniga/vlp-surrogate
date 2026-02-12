from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ____vfp.models import FileType

logger = logging.getLogger(__name__)


def get_mapping(file_type: FileType) -> dict[str, int]:
    if file_type == "p":
        mapping = {"time": 0, "flow": 2, "thp": 4, "wgr": 3, "bhp": 1}
    elif file_type == "i":
        mapping = {"time": 0, "flow": 2, "thp": 3, "bhp": 1}
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    logger.debug("Mapping selected", extra={"file_type": file_type, "mapping": mapping})
    return mapping


def read_xlsx(path: str | Path, file_type: FileType, start_line: int) -> np.ndarray:
    logger.info(
        "Reading XLSX",
        extra={
            "file_path": str(path),
            "file_type": file_type,
            "start_line": start_line,
        },
    )
    if file_type == "p":
        return read_production_xlsx(path, start_line)
    if file_type == "i":
        return read_injection_xlsx(path, start_line)
    raise ValueError(f"Unsupported file type: {file_type}")


def prepare_vfp_content(
    predicted_bhp: np.ndarray, n_size: int, *, include_wgr: bool
) -> np.ndarray:
    rows: list[list[float]] = []
    if include_wgr:
        ogr_index = 1
        lift_index = 1
        row_index = 0
        for wgr_index in range(1, n_size + 1):
            for thp_index in range(1, n_size + 1):
                rows.append(
                    [
                        thp_index,
                        wgr_index,
                        ogr_index,
                        lift_index,
                        *predicted_bhp[row_index],
                    ]
                )
                row_index += 1
        table = np.array(rows, dtype=float)
        logger.info(
            "Prepared VFP content",
            extra={"rows": int(table.shape[0]), "columns": int(table.shape[1])},
        )
        return table

    for thp_index in range(1, n_size + 1):
        rows.append([thp_index, *predicted_bhp[thp_index - 1]])
    table = np.array(rows, dtype=float)
    logger.info(
        "Prepared VFP content",
        extra={"rows": int(table.shape[0]), "columns": int(table.shape[1])},
    )
    return table


def read_production_xlsx(path: str | Path, start_line: int) -> np.ndarray:
    table = _read_xlsx(path, start_line, column_count=5)
    logger.info("Production XLSX loaded", extra={"rows": int(table.shape[0])})
    return table


def read_injection_xlsx(path: str | Path, start_line: int) -> np.ndarray:
    table = _read_xlsx(path, start_line, column_count=4)
    logger.info("Injection XLSX loaded", extra={"rows": int(table.shape[0])})
    return table


def _read_xlsx(path: str | Path, start_line: int, *, column_count: int) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading XLSX requires pandas and openpyxl. "
            "Install with `uv pip install pandas openpyxl`."
        ) from exc

    file_path = Path(path)
    frame = pd.read_excel(
        file_path,
        header=None,
        skiprows=start_line - 1,
        usecols=list(range(column_count)),
        engine="openpyxl",
    )
    frame = frame.fillna(0)
    data = frame.to_numpy(dtype=float)
    logger.debug(
        "XLSX loaded",
        extra={"rows": int(data.shape[0]), "columns": int(data.shape[1])},
    )
    return data
