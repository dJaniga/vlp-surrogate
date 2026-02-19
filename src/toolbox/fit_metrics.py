import numpy as np
import numpy.typing as npt


def all_fit_metrics(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    return {
        "rmse": rmse(y_actual, y_pred, mask),
        "mae": mae(y_actual, y_pred, mask),
        "r2": r2(y_actual, y_pred, mask),
    }


def rmse(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """Root Mean Squared Error; returns NaN if mask selects no finite values."""
    if mask is None:
        mask = np.ones_like(y_actual, dtype=bool)
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
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """Mean Absolute Error; returns NaN if mask selects no finite values."""
    if mask is None:
        mask = np.ones_like(y_actual, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return float("nan")
    diff = np.abs(y_actual[mask] - y_pred[mask])
    diff = diff[np.isfinite(diff)]
    return float(np.mean(diff)) if diff.size else float("nan")


def r2(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    if mask is None:
        mask = np.ones_like(y_actual, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return float("nan")
    diff = y_actual[mask] - y_pred[mask]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    return float(
        1
        - np.sum((y_actual[mask] - y_pred[mask]) ** 2)
        / np.sum((y_actual[mask] - np.mean(y_actual[mask])) ** 2)
    )
