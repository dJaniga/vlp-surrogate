import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vfp.modeling import VFPModel
from sklearn.linear_model import HuberRegressor as HBRegressor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HuberRegressor(VFPModel):
    _model: HBRegressor = field(default=None, init=False)
    epsilon: float = 1.35
    alpha: float = 0.0001
    seed: int | None = None

    def __str__(self) -> str:
        return "huber_regressor"

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
        logger.debug(
            "Fitting elastic net regression",
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

        self._model = HBRegressor(epsilon=self.epsilon, alpha=self.alpha, max_iter=200)
        self._model.fit(features, targets.ravel())
        logger.debug("Coefficients", extra={"coefficients": self.get_fit_details()})
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)
