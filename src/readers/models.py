from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from pandera import Timestamp


class ProductionFlowData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    WFR: float
    GFR: float
    BHP: float


class InjectionFlowData(pa.DataFrameModel):
    T: Timestamp
    FLO: float
    THP: float
    BHP: float


class FlowData(NamedTuple):
    production: pd.DataFrame | None = None
    injection: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class FitResults:
    overall: Mapping[str, float] = field(default_factory=dict)
    production: Mapping[str, float] = field(default_factory=dict)
    injection: Mapping[str, float] = field(default_factory=dict)


type WellName = str
type WellsFLowData = dict[WellName, FlowData]
type WellsFitResults = dict[WellName, FitResults]
type MetricFn = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    float,
]
