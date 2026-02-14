import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any

import numpy as np

from modeling.base import VFPModel
from models import (
    WellsData,
    WellsTrainingData,
    FlowData,
    TrainingData,
    VFPPipelineConfig,
)
from models.common import FlowType
from models.pipeline import WellsFittedModels, WellsPredictions, ModelPredictions, WellsVFPContent
from pipeline.transformers import to_features, to_prediction, reshape_predictions
from readers import ReaderProtocol, EclipseReader

logger = logging.getLogger(__name__)

READER_MAPPING: dict[str, type[ReaderProtocol]] = {
    ".unsmry": EclipseReader,
}


@dataclass(slots=True, frozen=True)
class VFPPipeline:
    config: VFPPipelineConfig
    reference_vfp_model: VFPModel

    def run(self) -> None:
        wells_data = _read_wells_data(self.config.input_file_path)
        cleaned_wells_data = _clean_wells_data(wells_data)
        wells_training_data = _transform_to_feature_and_targets(
            cleaned_wells_data,
            self.config.production_feature_keys,
            self.config.injection_feature_keys,
            self.config.target_keys,
        )
        wells_fitted_models = _fit_models(wells_training_data,
                                          self.reference_vfp_model)
        wells_predicted_data = _predict_models(wells_fitted_models, cleaned_wells_data,
                                               self.config.vfp_table_granularity,
                                               self.config.production_feature_keys,
                                               self.config.injection_feature_keys)
        wells_vfp_content = _prepare_vfp_content(wells_predicted_data,
                                                 self.config.vfp_table_granularity)


def _read_wells_data(input_file_path: Path) -> WellsData:
    logger.info("Reading training data from %s", input_file_path)
    suffix = input_file_path.suffix.lower()
    try:
        reader: type[ReaderProtocol] = READER_MAPPING[suffix]
    except KeyError as e:
        raise ValueError(f"Unsupported file extension: {suffix!r}") from e
    return reader.read_wells_data(input_file_path)


def _clean_wells_data(training_data: WellsData) -> WellsData:
    return training_data


def _transform_to_feature_and_targets(
        wells_data: WellsData,
        production_feature_keys: tuple[str, ...],
        injection_feature_keys: tuple[str, ...],
        target_keys: tuple[str, ...],
) -> WellsTrainingData:
    wells_training_data: WellsTrainingData = {}

    def _get_one(
            flow_data: FlowData | None,
            feature_keys: Sequence[str],
            target_keys: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        f = (
            to_features(flow_data, keys=list(feature_keys))
            if flow_data is not None
            else None
        )
        t = (
            to_features(flow_data, keys=list(target_keys))
            if flow_data is not None
            else None
        )
        return f, t

    for w_name, (production_data, injection_data) in wells_data.items():
        prod_f, prod_t = _get_one(production_data, production_feature_keys, target_keys)
        inj_f, inj_t = _get_one(injection_data, injection_feature_keys, target_keys)
        wells_training_data[w_name] = FlowType[TrainingData](
            production=TrainingData(features=prod_f, targets=prod_t),
            injection=TrainingData(features=inj_f, targets=inj_t),
        )

    return wells_training_data


def _fit_models(
        wells_training_data: WellsTrainingData, reference_model: VFPModel
) -> WellsFittedModels:
    wells_fitted_models: WellsFittedModels = {}

    def _fit_one(training_data: TrainingData | None, reference_model: VFPModel) -> VFPModel | None:
        if training_data is None:
            return None
        if training_data.features is None or training_data.targets is None:
            return None
        model = copy.deepcopy(reference_model)
        model.fit(training_data.features, training_data.targets)
        return model

    for w_name, (prod, inj) in wells_training_data.items():
        prod_m = _fit_one(prod, reference_model)
        inj_m = _fit_one(inj, reference_model)
        wells_fitted_models[w_name] = FlowType[VFPModel](production=prod_m, injection=inj_m)

    return wells_fitted_models


def _predict_models(wells_fitted_models: WellsFittedModels,
                    cleaned_wells_data: WellsData,
                    n_size: int,
                    production_feature_keys: tuple[str, ...],
                    injection_feature_keys: tuple[str, ...],
                    ) -> WellsPredictions:
    wells_prediction: WellsPredictions = {}

    def _predict_one(model: VFPModel | None,
                     flow_data: Any | None,
                     feature_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if model is None or flow_data is None:
            return None, None
        prediction_features = to_prediction(flow_data, keys=list(feature_keys), n_size=n_size)
        predicted_target = model.predict(prediction_features)
        reshaped_predicted_target = reshape_predictions(predicted_target, n_size)
        return prediction_features, reshaped_predicted_target

    for w_name, (prod, inj) in cleaned_wells_data.items():
        prod_p_f, r_prod_t = _predict_one(wells_fitted_models[w_name].production, prod, production_feature_keys)
        inj_p_f, r_inj_t = _predict_one(wells_fitted_models[w_name].injection, inj, injection_feature_keys)
        wells_prediction[w_name] = FlowType[ModelPredictions](
            production=ModelPredictions(prediction_features=prod_p_f, predicted_target=r_prod_t),
            injection=ModelPredictions(prediction_features=inj_p_f, predicted_target=r_inj_t)
        )

    return wells_prediction


def _prepare_vfp_content(wells_predictions: WellsPredictions,
                         n_size: int) -> WellsVFPContent:
    wells_vfp_content: WellsVFPContent = {}

    def _generate_records_indexes(features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError("features must be a 2D array")

        # Count unique values per column (excluding column with flow data)
        unique_counts = [
            np.unique(features[:, col]).size
            for col in range(1, features.shape[1])
        ]

        # Add ALQ index if more than one column exists -> production tables
        if features.shape[1] > 1:
            unique_counts.append(1)

        if not unique_counts:
            return np.empty((0, 0), dtype=int)

        grids = np.meshgrid(
            *[np.arange(1, n + 1) for n in unique_counts],
            indexing="xy"
        )

        return np.stack(grids, axis=-1).reshape(-1, len(unique_counts))

    def _prepare_one(prediction_features: np.ndarray | None, predicted_target: np.ndarray | None) -> np.ndarray | None:
        if prediction_features is None or predicted_target is None:
            return None
        records_indexes = _generate_records_indexes(prediction_features)

        print(records_indexes)

    for w_name, (prod, inj) in wells_predictions.items():
        prod_c = _prepare_one(prod.prediction_features, prod.predicted_target)
        inj_c = _prepare_one(inj.prediction_features, inj.predicted_target)

    return wells_vfp_content
