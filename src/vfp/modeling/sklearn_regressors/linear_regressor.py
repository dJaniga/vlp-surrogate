from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from vfp.modeling.base import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LinearRegressor(VFPModel):
    """Simple linear regression using least squares."""

    coefficients: np.ndarray | None = None
    seed: int | None = None  # Only for API consistency

    def get_fit_details(self) -> dict[str, Any]:
        if self.coefficients is None:
            raise ValueError("Model has not been fit yet.")
        return dict(zip(self.features_name, self.coefficients))

    def __str__(self) -> str:
        return "linear_regressor"

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> LinearRegressor:
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
        self.features_name = ("Intercept",) + (
            features_name
            if features_name
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        logger.info("Coefficients", extra={"coefficients": self.get_fit_details()})

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
