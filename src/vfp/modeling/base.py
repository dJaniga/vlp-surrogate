from __future__ import annotations

from typing import Protocol

import numpy as np


class VFPModel(Protocol):
    def fit(self, features: np.ndarray, targets: np.ndarray, features_name: tuple[str,...] | None = None) -> VFPModel: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...
