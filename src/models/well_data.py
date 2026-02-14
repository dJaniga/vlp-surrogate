from dataclasses import dataclass, field

import pandera.pandas as pa
from pandera import Timestamp


## TODO: can production and injection be combined on single model?
class ProductionData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    WFR: float
    GFR: float
    BHP: float


class InjectionData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    BHP: float


@dataclass(frozen=True, slots=True)
class FlowData:
    production: ProductionData | None = field(default=None)
    injection: InjectionData | None = field(default=None)

    def __iter__(self):
        yield from (self.production, self.injection)


type WellsData = dict[str, FlowData]
