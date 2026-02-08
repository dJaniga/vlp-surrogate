from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vfp.preprocess import (
    clean_dataset,
    read_xlsx,
    get_mapping,
    prepare_data_for_model,
    prepare_data_for_prediction,
    prepare_vfp_content,
)
import logging

from vfp.models import VFPConfig, VFPDataConfig, VFPTableConfig
from vfp.utils import save_to_vfp_file
from vfp.modeling import LinearRegressionModel, VFPModel

logger = logging.getLogger(__name__)


def reshape_predictions(predictions: np.ndarray, n_size: int) -> np.ndarray:
    if predictions.size % n_size != 0:
        raise ValueError("Prediction count is not divisible by grid size.")
    return predictions.reshape((-1, n_size)).round(3)


@dataclass(slots=True)
class VFPTableResult:
    prediction_data: np.ndarray
    predicted_bhp: np.ndarray
    table_content: np.ndarray


@dataclass(slots=True)
class VFPTablePipeline:
    data_config: VFPDataConfig
    table_config: VFPTableConfig
    model: VFPModel | None = None

    def run(self, output_path: str | Path) -> VFPTableResult:
        logger.info(
            "Loading dataset",
            extra={
                "file_path": str(self.data_config.file_path),
                "file_type": self.data_config.file_type,
            },
        )
        mapping = get_mapping(self.data_config.file_type)
        dataset = read_xlsx(
            self.data_config.file_path,
            self.data_config.file_type,
            self.data_config.data_start_line,
        )
        cleaned = clean_dataset(
            dataset,
            self.data_config.minimum_acceptable_flow,
            self.data_config.start_time,
            mapping,
        )
        logger.info(
            "Dataset cleaned",
            extra={"rows": int(cleaned.shape[0]), "columns": int(cleaned.shape[1])},
        )
        training_data = prepare_data_for_model(cleaned, mapping)
        if training_data.size == 0:
            raise ValueError("No training data available after preprocessing.")
        features = training_data[:, :-1]
        targets = training_data[:, -1]

        model = self.model or LinearRegressionModel()
        logger.info(
            "Training model",
            extra={
                "model": model.__class__.__name__,
                "samples": int(features.shape[0]),
            },
        )
        model.fit(features, targets)

        prediction_data = prepare_data_for_prediction(
            cleaned, self.table_config.vfp_parameter_size, mapping
        )
        predictions = model.predict(prediction_data)
        predicted_bhp = reshape_predictions(
            predictions, self.table_config.vfp_parameter_size
        )

        table_content = prepare_vfp_content(
            predicted_bhp,
            self.table_config.vfp_parameter_size,
            include_wgr="wgr" in mapping,
        )

        save_to_vfp_file(table_content, prediction_data, output_path, self.table_config)
        logger.info(
            "VFP table written",
            extra={"output": str(output_path), "rows": int(table_content.shape[0])},
        )
        return VFPTableResult(
            prediction_data=prediction_data,
            predicted_bhp=predicted_bhp,
            table_content=table_content,
        )


def run_pipeline(config: VFPConfig) -> VFPTableResult:
    pipeline = VFPTablePipeline(config.data, config.table)
    return pipeline.run(config.output_path)
