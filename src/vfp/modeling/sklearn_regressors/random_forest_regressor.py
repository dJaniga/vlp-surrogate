from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor

from vfp.modeling import VFPModel


@dataclass(slots=True)
class RandomForestRegressor(VFPModel):
    _model: SKRandomForestRegressor = field(default=None, init=False)
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.4,
    max_features = 1

    seed: int | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray, features_name: tuple[str, ...] | None = None,
            eval_set: tuple[np.ndarray, np.ndarray] | None = None) -> VFPModel:
        self._model = SKRandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=-1,
        )

        self._model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def __str__(self) -> str:
        return "random_forest_regressor"

    def get_fit_details(self) -> dict[str, Any]:
        return {}
