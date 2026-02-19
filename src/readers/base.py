from abc import ABC, abstractmethod
from pathlib import Path

from readers.models import WellsFLowData, WellsFitResults, WellDataFilter


class ReaderInterface(ABC):
    @classmethod
    @abstractmethod
    def read_wells_flow_data(
        cls,
        ecl_smr_file_path: str | Path,
        well_data_filter: WellDataFilter | None = None,
    ) -> WellsFLowData: ...

    @classmethod
    @abstractmethod
    def calculate_wells_fits(cls, ecl_smr_file_path: str | Path) -> WellsFitResults: ...
