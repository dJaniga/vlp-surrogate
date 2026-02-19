from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd


class PredictionContent(NamedTuple):
    features: np.ndarray
    records_config: pd.DataFrame


def to_prediction_format(
    df: pd.DataFrame, keys: Sequence[str], n_size: int
) -> PredictionContent:
    reversed_keys = list(reversed(keys))

    grids = np.asarray(
        [np.linspace(df[k].min(), df[k].max(), n_size) for k in reversed_keys],
        dtype=float,
    )

    at_least_one_unique = np.asarray(
        [np.any(r > 0) for r in grids],
        dtype=bool,
    )

    active_grids = grids[at_least_one_unique]
    mesh = np.meshgrid(*active_grids, indexing="ij")
    stacked = np.stack(mesh, axis=-1)
    arr_reduced = stacked.reshape(-1, np.sum(at_least_one_unique))

    n_total_dims = len(keys)
    n_rows = arr_reduced.shape[0]
    arr_full = np.zeros((n_rows, n_total_dims), dtype=float)

    arr_full[:, at_least_one_unique] = arr_reduced

    arr_in_original_order = arr_full[:, ::-1]

    records_config = pd.DataFrame(data=grids.T, columns=reversed_keys)[keys]
    if {"WFR", "GFR"} & set(records_config.columns):
        records_config["ALQ"] = 0.0

    return PredictionContent(
        features=arr_in_original_order, records_config=records_config
    )


def reshape_predictions(predictions: np.ndarray, n_size: int) -> np.ndarray:
    if predictions.size % n_size != 0:
        raise ValueError("Prediction count is not divisible by grid size.")
    return predictions.reshape((-1, n_size)).round(5)
