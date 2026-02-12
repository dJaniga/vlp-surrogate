from abc import ABC, abstractmethod
from pathlib import Path

from models import VLPTrainingData


class ReaderProtocol(ABC):

    @classmethod
    @abstractmethod
    def prepare_training_data(cls, file_path: Path, **kwargs) -> list[VLPTrainingData]:
        ...
