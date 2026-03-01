from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from vfp.modeling.base import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class XGBoostRegressor(VFPModel):
    """Simple linear regression using least squares."""

    _model: XGBRegressor | None = field(default=None, init=False)
    xgb_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_fit_details(self) -> dict[str, Any]:
        if self._model is None:
            raise ValueError("Model has not been fit yet.")
        # xgboost provides feature importances
        importances = self._model.feature_importances_
        if self.features_name is None:
            return {}
        return dict(zip(self.features_name, importances))

    def __str__(self) -> str:
        return "xgb_regressor"

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> XGBoostRegressor:
        self.features_name = (
            features_name
            if features_name
            else {f"ARG{idx}": name for idx, name in enumerate(features_name)}
        )

        logger.info(
            "Fitting XGBoost regression",
            extra={
                "samples": int(features.shape[0]),
                "features": int(features.shape[1]),
                "hyperparameters": self.xgb_kwargs,
            },
        )

        # Inject early stopping if eval_set is provided
        kwargs = self.xgb_kwargs.copy()
        if eval_set is not None and "early_stopping_rounds" not in kwargs:
            kwargs["early_stopping_rounds"] = 10

        self._model = XGBRegressor(**kwargs)
        if eval_set is not None:
            self._model.fit(
                features,
                targets,
                eval_set=[eval_set],
                verbose=0,
            )
            if hasattr(self._model, "best_iteration"):
                logger.info(
                    "XGBoost early stopping",
                    extra={"best_iteration": getattr(self._model, "best_iteration")},
                )
        else:
            self._model.fit(features, targets)

        logger.info(
            "Feature Importances", extra={"importances": self.get_fit_details()}
        )

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model has not been fit yet.")
        logger.info(
            "Predicting with XGB regression",
            extra={"samples": int(features.shape[0])},
        )
        return self._model.predict(features)
