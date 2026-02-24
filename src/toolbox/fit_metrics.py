import inspect

import numpy as np
import numpy.typing as npt
import sklearn.metrics as skm


def run_all_regression_metrics(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_] | None = None,
    *,
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """
    Compute all sklearn regression metrics with masking and NaN safety.

    Returns NaN for metrics if no valid samples remain.
    """

    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if mask is None:
        mask = np.ones_like(y_actual, dtype=bool)

    mask = np.asarray(mask, dtype=bool)

    # Basic validity mask
    valid = mask & np.isfinite(y_actual) & np.isfinite(y_pred)

    if valid.size == 0 or not np.any(valid):
        return {}

    y_true_valid = y_actual[valid]
    y_pred_valid = y_pred[valid]

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)[valid]

    results: dict[str, float] = {}
    metrics = {
        "d2_absolute_error_score",
        "d2_pinball_score",
        "d2_tweedie_score",
        "explained_variance_score",
        "max_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_gamma_deviance",
        "mean_pinball_loss",
        "mean_poisson_deviance",
        "mean_squared_error",
        "mean_squared_log_error",
        "mean_tweedie_deviance",
        "median_absolute_error",
        "r2_score",
        "root_mean_squared_error",
        "root_mean_squared_log_error",
    }

    for name, func in inspect.getmembers(skm, inspect.isfunction):
        if name not in metrics:
            continue

        try:
            sig = inspect.signature(func)

            kwargs = {}
            if "sample_weight" in sig.parameters and sample_weight is not None:
                kwargs["sample_weight"] = sample_weight

            value = func(y_true_valid, y_pred_valid, **kwargs)

            # Force scalar float output
            if np.ndim(value) > 0:
                value = float(np.mean(value))

            results[name] = float(value)

        except Exception:
            results[name] = float("nan")

    return results
