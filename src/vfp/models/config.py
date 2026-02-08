from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FileType = Literal["p", "i"]
UnitSystem = Literal["METRIC", "FIELD", "LAB", "PVT-M"]


@dataclass(frozen=True, slots=True)
class VFPDataConfig:
    file_path: Path
    file_type: FileType
    data_start_line: int = 5
    minimum_acceptable_flow: float = 500.0
    start_time: float = 0.0


@dataclass(frozen=True, slots=True)
class VFPTableConfig:
    vfp_parameter_size: int = 5
    vlp_table_number: int = 1
    well_depth: float = 0.0
    unit_system: UnitSystem = "METRIC"
    flow_definition: str = "GAS"
    thp_definition: str = "THP"
    wfr_definition: str = "WGR"
    gfr_definition: str = "OGR"
    alq_definition: str = "GRAT"
    tabulated_quantity: str = "BHP"


@dataclass(frozen=True, slots=True)
class VFPConfig:
    data: VFPDataConfig
    table: VFPTableConfig
    output_path: Path
