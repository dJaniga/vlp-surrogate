import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.svm import SVR

from vfp.modeling import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SVRRegressor(VFPModel):
    _model: SVR = field(default=None, init=False)
    kernel: str = "rbf"
    degree: int = 3
    C: float = 1.0
    epsilon: float = 0.1
    seed: int | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> VFPModel:
        logger.debug(
            "Fitting SVR regression",
        )
        self.features_name = ("Intercept",) + (
            features_name
            if features_name
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        self._model = SVR(
            kernel=self.kernel, degree=self.degree, C=self.C, epsilon=self.epsilon
        )

        self._model.fit(features, targets.ravel())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def __str__(self) -> str:
        return "svr_regressor"

    def get_fit_details(self) -> dict[str, Any]:
        return {}
