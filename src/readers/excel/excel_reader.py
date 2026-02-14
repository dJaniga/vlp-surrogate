from pathlib import Path

from readers.models import VLPType, VLPTrainingData


class ExcelReader:
    @classmethod
    def prepare_training_data(
        cls,
        file_path: str | Path,
        well_name: str | None = None,
        vlp_type: VLPType = VLPType.BOTH,
    ) -> VLPTrainingData | list[VLPTrainingData]:

        raise NotImplementedError
