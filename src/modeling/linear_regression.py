from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from modeling.base import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LinearRegressionModel(VFPModel):
    """Simple linear regression using least squares."""

    coefficients: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> LinearRegressionModel:
        logger.info(
            "Fitting linear regression",
            extra={
                "samples": int(features.shape[0]),
                "features": int(features.shape[1]),
            },
        )
        design_matrix = np.column_stack([np.ones(features.shape[0]), features])
        coefficients, *_ = np.linalg.lstsq(design_matrix, targets, rcond=None)
        self.coefficients = coefficients
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model has not been fit yet.")
        logger.info(
            "Predicting with linear regression",
            extra={"samples": int(features.shape[0])},
        )
        design_matrix = np.column_stack([np.ones(features.shape[0]), features])
        return design_matrix @ self.coefficients
