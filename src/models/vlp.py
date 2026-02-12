from dataclasses import dataclass, field

import pandera.pandas as pa
from pandera import Timestamp


class VLPPData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    WFR: float
    GFR: float
    BHP: float


class VLPIData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    BHP: float


@dataclass(frozen=True, slots=True)
class VLPTrainingData:
    well_name: str
    production: VLPPData | None = field(default=None)
    injection: VLPIData | None = field(default=None)
