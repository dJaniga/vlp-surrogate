from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from vfp.modeling.base import VFPModel

logger = logging.getLogger(__name__)

_EARLY_STOPPING_DEFAULT = 100
_FIT_METRIC: str = os.environ.get("VLP_FIT_METRIC", "mse")


@dataclass(slots=True)
class XGBoostRegressor(VFPModel):
    """XGBoost-based regression model with optional early stopping."""

    _model: XGBRegressor | None = field(default=None, init=False)
    xgb_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    @property
    def _is_fitted(self) -> bool:
        return self._model is not None

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("Model has not been fit yet.")

    def get_fit_details(self) -> dict[str, Any]:
        self._require_fitted()
        if not self.features_name:
            return {}
        return dict(zip(self.features_name, self._model.feature_importances_))

    def __str__(self) -> str:
        return "xgb_regressor"

    def _build_kwargs(
        self, eval_set: tuple[np.ndarray, np.ndarray] | None
    ) -> dict[str, Any]:
        kwargs = self.xgb_kwargs.copy()
        if eval_set is not None:
            kwargs.setdefault("early_stopping_rounds", _EARLY_STOPPING_DEFAULT)
        else:
            kwargs.pop("early_stopping_rounds", None)
        if self.seed is not None:
            kwargs.setdefault("random_state", self.seed)
        return kwargs

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> XGBoostRegressor:
        self.features_name = features_name or tuple(
            f"ARG{i}" for i in range(features.shape[1])
        )

        kwargs = self._build_kwargs(eval_set)

        logger.debug(
            "Fitting XGBoost regression",
            extra={
                "samples": int(features.shape[0]),
                "features": int(features.shape[1]),
                "hyperparameters": kwargs,
            },
        )

        self._model = XGBRegressor(**kwargs)
        self._model.fit(
            features,
            targets,
            eval_set=[eval_set] if eval_set is not None else None,
            verbose=False,
        )

        if eval_set is not None and hasattr(self._model, "best_iteration"):
            logger.debug(
                "XGBoost early stopping",
                extra={"best_iteration": self._model.best_iteration},
            )

        logger.debug(
            "Feature importances", extra={"importances": self.get_fit_details()}
        )

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        logger.info(
            "Predicting with XGBoost regression",
            extra={"samples": int(features.shape[0])},
        )
        return self._model.predict(features)
