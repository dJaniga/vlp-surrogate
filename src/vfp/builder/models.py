from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class VFPTable:
    header: str
    config_records: pd.DataFrame  # config records
    body: np.ndarray  # indexes with bhp


class VFPType(StrEnum):
    PRODUCTION = "VFPPROD"
    INJECTION = "VFPINJ"
