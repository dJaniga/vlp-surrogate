import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from modeling import VFPModel, create_model
from models import ModelName, VLPTrainingData
from pipeline.transformers import to_features, to_targets
from readers import EclipseReader, ReaderProtocol
from utils import ensure_not_none


@dataclass(frozen=True, slots=True)
class _FittedVfpModels:
    production: VFPModel | None
    injection: VFPModel | None


class Builder:
    _READER_MAPPING: dict[str, type[ReaderProtocol]] = {
        ".unsmry": EclipseReader,
    }

    _PROD_FEATURE_KEYS = ("FLO", "THP", "WFR", "GFR")
    _INJ_FEATURE_KEYS = ("FLO", "THP")
    _TARGET_KEYS = ("BHP",)

    def __init__(self, file_path: Path):
        self._file_path: Path = file_path

        suffix = self._file_path.suffix.lower()
        try:
            self._reader: type[ReaderProtocol] = Builder._READER_MAPPING[suffix]
        except KeyError as e:
            raise ValueError(f"Unsupported file extension: {suffix!r}") from e

        self._wells_training_data: list[VLPTrainingData] | None = None
        self._reference_vfp_model: VFPModel | None = None
        self._fitted_vfp_models: dict[str, _FittedVfpModels] = {}

    def read(self) -> Self:
        self._wells_training_data = self._reader.prepare_training_data(self._file_path)
        return self

    def create_new_vfp_model(self, model_name: ModelName, **kwargs) -> Self:
        if self._wells_training_data is None:
            raise RuntimeError("Call read() before create_new_vfp_model().")

        self._reference_vfp_model = create_model(model_name, **kwargs)
        self._fit_models()
        self._predict_on_models()
        return self

    def save_vfp(self, output_folder_path: Path) -> None:
        pass

    def _fit_models(self) -> None:
        ref_model = ensure_not_none(self._reference_vfp_model)

        def _fit_one(training, feature_keys: tuple[str, ...]) -> VFPModel:
            model = copy.deepcopy(ref_model)
            features = to_features(training, keys=list(feature_keys))
            targets = to_targets(training, keys=list(self._TARGET_KEYS))
            model.fit(features, targets)
            return model

        for w in ensure_not_none(self._wells_training_data):
            production_model = (
                _fit_one(w.production, self._PROD_FEATURE_KEYS) if w.production is not None else None
            )
            injection_model = (
                _fit_one(w.injection, self._INJ_FEATURE_KEYS) if w.injection is not None else None
            )
            self._fitted_vfp_models[w.well_name] = _FittedVfpModels(production_model, injection_model)

    def _predict_on_models(self) -> None:
        pass
