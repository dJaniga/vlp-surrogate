from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from modeling import (
    GeneticAlgorithmRegressor,
    LinearRegressionModel,
    SymbolicRegressor,
    VFPModel,
)
from ____vfp.models import VFPDataConfig
from ____vfp.preprocess import clean_dataset, get_mapping, prepare_data_for_model, read_xlsx

logger = logging.getLogger(__name__)

ModelName = Literal["linear", "ga", "symbolic"]


@dataclass(slots=True)
class VFPPreparedData:
    """Container for prepared training arrays."""

    mapping: dict[str, int]
    cleaned: np.ndarray
    features: np.ndarray
    targets: np.ndarray


def load_training_data(config: VFPDataConfig) -> VFPPreparedData:
    """Load, clean, and reshape training data from an XLSX source."""
    logger.info(
        "Loading training data",
        extra={"file_path": str(config.file_path), "file_type": config.file_type},
    )
    mapping = get_mapping(config.file_type)
    dataset = read_xlsx(config.file_path, config.file_type, config.data_start_line)
    cleaned = clean_dataset(
        dataset,
        config.minimum_acceptable_flow,
        config.start_time,
        mapping,
    )
    training_data = prepare_data_for_model(cleaned, mapping)
    if training_data.size == 0:
        raise ValueError("No training data available after preprocessing.")

    features = training_data[:, :-1]
    targets = training_data[:, -1]
    prepared = VFPPreparedData(
        mapping=mapping,
        cleaned=cleaned,
        features=features,
        targets=targets,
    )
    logger.info(
        "Training data ready",
        extra={
            "rows": int(prepared.features.shape[0]),
            "columns": int(prepared.features.shape[1]),
        },
    )
    return prepared


def create_model(name: ModelName, **kwargs) -> VFPModel:
    """Factory for supported models (linear or ga)."""
    logger.info("Creating model", extra={"model": name})
    if name == "linear":
        return LinearRegressionModel(**kwargs)
    if name == "ga":
        return GeneticAlgorithmRegressor(**kwargs)
    if name == "symbolic":
        return SymbolicRegressor(**kwargs)
    raise ValueError(f"Unsupported model name: {name}")


def fit_model(data: VFPPreparedData, model: VFPModel | None = None) -> VFPModel:
    """Fit a model using prepared data."""
    model = model or LinearRegressionModel()
    logger.info(
        "Fitting model",
        extra={
            "model": model.__class__.__name__,
            "samples": int(data.features.shape[0]),
        },
    )
    model.fit(data.features, data.targets)
    return model
