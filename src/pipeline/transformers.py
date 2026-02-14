import logging
from collections.abc import Sequence
from typing import Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_features(df: Type[pd.DataFrame], keys: Sequence[str]) -> np.ndarray:
    f = df[keys].to_numpy()
    at_least_one_unique = np.any(f > 0, axis=0)
    if not np.all(at_least_one_unique):
        logger.warning(f"All data for {np.asarray(keys)[~at_least_one_unique]} have zero value, excluding from training features ")

    return f[:, at_least_one_unique]


def to_targets(df: Type[pd.DataFrame], keys: Sequence[str]) -> np.ndarray:
    return df[keys].to_numpy()


def to_prediction(df: pd.DataFrame, keys: Sequence[str], n_size: int) -> np.ndarray:
    reversed_keys = list(reversed(keys))
    grids = np.asarray([np.linspace(df[k].min(), df[k].max(), n_size) for k in reversed_keys], dtype=float)
    at_least_one_unique = np.asarray([np.any(r > 0) for r in grids], dtype=bool)
    mesh = np.meshgrid(*grids[at_least_one_unique], indexing="ij")
    stacked = np.stack(mesh, axis=-1)
    arr = stacked.reshape(-1, np.sum(at_least_one_unique))
    return arr[:, ::-1].astype(float)


def reshape_predictions(predictions: np.ndarray, n_size: int) -> np.ndarray:
    if predictions.size % n_size != 0:
        raise ValueError("Prediction count is not divisible by grid size.")
    return predictions.reshape((-1, n_size)).round(3)
