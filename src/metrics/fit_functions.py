from typing import Callable

import numpy as np
import numpy.typing as npt

type MetricFn = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]],
    float,
]

def rmse(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> float:
    """Root Mean Squared Error; returns NaN if mask selects no finite values."""
    if mask.size == 0 or not np.any(mask):
        return float("nan")
    diff = y_actual[mask] - y_pred[mask]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(diff))))


def mae(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> float:
    if mask.size == 0 or not np.any(mask):
        return float("nan")
    diff = np.abs(y_actual[mask] - y_pred[mask])
    diff = diff[np.isfinite(diff)]
    return float(np.mean(diff)) if diff.size else float("nan")
