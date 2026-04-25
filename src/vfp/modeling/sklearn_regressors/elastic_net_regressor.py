import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet

from vfp.modeling import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ElasticNetRegressor(VFPModel):
    _model: ElasticNet = field(default=None, init=False)
    alpha: float = 1.0
    l1_ratio: float = 0.5
    seed: int | None = None

    def __str__(self) -> str:
        return "elastic_net_regressor"

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
    ) -> ElasticNetRegressor:
        logger.info(
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
        self._model = ElasticNet(
            random_state=self.seed, alpha=self.alpha, l1_ratio=self.l1_ratio
        )
        self._model.fit(features, targets)

        logger.info("Coefficients", extra={"coefficients": self.get_fit_details()})

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)
