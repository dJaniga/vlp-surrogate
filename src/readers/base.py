from abc import ABC, abstractmethod
from pathlib import Path

from readers.models import (
    WellsFlowData,
    WellsFitResults,
    WellDataFilter,
)


class ReaderInterface(ABC):
    @classmethod
    @abstractmethod
    def from_file(cls, path: str | Path) -> ReaderInterface: ...

    @abstractmethod
    def read_wells_flow_data(
        self,
        well_data_filter: WellDataFilter | None = None,
    ) -> WellsFlowData:
        """
        Return per-well flow datasets.
        Reader instance is assumed to already be initialized
        with required data source (e.g., Summary).
        """
        ...

    @abstractmethod
    def calculate_wells_fits(
        self,
        well_data_filter: WellDataFilter | None = None,
    ) -> WellsFitResults:
        """
        Compute per-well fit metrics.
        Reader instance is assumed to already be initialized
        with required data source (e.g., Summary).
        """
        ...
