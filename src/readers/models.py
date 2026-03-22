from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple, Counter

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
from pandera import Timestamp
from pydantic import BaseModel, model_validator, Field


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
    data_frame: pd.DataFrame = field(default_factory=pd.DataFrame)


class TimeFilter(BaseModel):
    from_T: datetime | None = Field(default=None, alias="from")
    to_T: datetime | None = Field(default=None, alias="to")

    @model_validator(mode="after")
    def validate_dates(self):
        if self.from_T and self.to_T and self.from_T > self.to_T:
            raise ValueError("from_T must be before to_T")
        return self


class WellDataFilter(BaseModel):
    wells: list[str] | None = Field(default=None)
    time: TimeFilter | None = Field(default=None)

    @model_validator(mode="after")
    def wells_must_be_not_empty(self):
        if self.wells is not None and not self.wells:
            raise ValueError("Wells filter must be not empty if provided.")
        return self

    @model_validator(mode="after")
    def well_must_have_unique_names(self):
        if not self.wells:
            return self

        counts = Counter(self.wells)
        duplicates = [name for name, count in counts.items() if count > 1]

        if duplicates:
            raise ValueError(f"Well names must be unique. Duplicates: {duplicates}")

        return self


type WellName = str
type WellsFlowData = dict[WellName, FlowData]
type WellsFitResults = dict[WellName, FitResults]
type MetricFn = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    float,
]
