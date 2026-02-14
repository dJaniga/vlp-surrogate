from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ModelName = Literal["linear", "ga", "symbolic"]


@dataclass(frozen=True, slots=True)
class VFPPipelineConfig:
    input_file_path: Path
    output_folder_path: Path
    vfp_table_granularity: int = 5
    production_feature_keys: tuple[str, ...] = ("FLO", "THP", "WFR", "GFR")
    injection_feature_keys: tuple[str, ...] = ("FLO", "THP")
    target_keys: tuple[str, ...] = ("BHP",)
