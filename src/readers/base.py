from abc import ABC, abstractmethod
from pathlib import Path

from models import WellsData


class ReaderProtocol(ABC):
    @classmethod
    @abstractmethod
    def read_wells_data(cls, file_path: Path, **kwargs) -> WellsData: ...
