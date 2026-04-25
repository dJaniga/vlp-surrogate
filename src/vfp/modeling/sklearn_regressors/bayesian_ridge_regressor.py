import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import BayesianRidge

from vfp.modeling import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BayesianRidgeRegressor(VFPModel):
    _model: BayesianRidge = field(default=None, init=False)
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    seed: int | None = None

    def __str__(self) -> str:
        return "bayesian_ridge_regressor"

    def get_fit_details(self) -> dict[str, Any]:
        if self._model is None:
            raise ValueError("Model has not been fit yet.")
        intercept = {self.features_name[0]: self._model.intercept_}
        coef = dict(zip(self.features_name[1:], self._model.coef_))
        return {**intercept, **coef}

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> VFPModel:
        logger.info(
            "Fitting Bayesian Ridge regression",
            extra={
                "samples": int(features.shape[0]),
                "features": int(features.shape[1]),
            },
        )

        self.features_name = ("Intercept",) + (
            features_name
            if features_name
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        if self.seed is not None:
            logger.warning(
                "Setting seed for Bayesian Ridge regression is not supported."
            )

        self._model = BayesianRidge(
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
        )
        self._model.fit(features, targets.ravel())

        logger.info("Coefficients", extra={"coefficients": self.get_fit_details()})

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)
