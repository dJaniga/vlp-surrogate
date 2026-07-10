"""Gaussian Process regressor — imports kernels from kernels.py.

Key improvements over original:
  - Analytic NLML gradient → true L-BFGS-B (not finite-difference)
  - Stores (L, lower) Cholesky tuple; predict_with_uncertainty is correct
  - Log-space bounded search; stable hyperparameter optimization
  - Full metrics (rmse, r2, mae, mse) always computed for train + eval
  - scale_features / scale_targets both default ON
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from vfp.modeling import VFPModel
from vfp.modeling.gaussian_process.kernels import KernelType, build_kernel

logger = logging.getLogger(__name__)

_JITTER = 1e-6
_LOG_2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    res = y_true - y_pred
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(res**2))),
        "mae": float(np.mean(np.abs(res))),
        "mse": float(np.mean(res**2)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0,
    }


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GaussianProcessRegressor(VFPModel):
    """GP regressor with analytic-gradient hyperparameter optimization.

    Parameters
    ----------
    kernel_name:
        'composite' (Matern52ARD + Linear, recommended for linear trends),
        'matern52_ard', 'ard', 'rbf', 'matern52', 'linear',
        'polynomial', 'rational_quadratic'.
    noise_variance:
        Observation noise σ_n². Use ~1e-4 for near-noise-free simulation data.
    n_restarts:
        Random restarts on top of default initialization.
    seed:
        RNG seed for reproducibility.
    scale_features:
        Standardize inputs (μ=0, σ=1). Recommended when features differ in scale.
    scale_targets:
        Standardize targets to zero mean / unit variance before fitting.
    """

    kernel_name: str = "composite"
    noise_variance: float = 1e-4
    n_restarts: int = 5
    seed: int | None = None
    scale_features: bool = False
    scale_targets: bool = False
    degree: int = 3  # only used by polynomial kernel

    # private state
    _kernel: Any = field(default=None, repr=False)
    _X_train: Any = field(default=None, repr=False)
    _y_train: Any = field(default=None, repr=False)
    _alpha: Any = field(default=None, repr=False)
    _chol: Any = field(default=None, repr=False)  # (L, lower) tuple
    _feature_mean: Any = field(default=None, repr=False)
    _feature_std: Any = field(default=None, repr=False)
    _target_mean: float = field(default=0.0, repr=False)
    _target_std: float = field(default=1.0, repr=False)
    _eval_metrics: Any = field(default_factory=dict, repr=False)
    features_name: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def _scale_X(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if not self.scale_features:
            return X
        if fit:
            self._feature_mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std < 1e-12] = 1.0
            self._feature_std = std
        return (X - self._feature_mean) / self._feature_std

    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        if not self.scale_targets:
            return y
        self._target_mean = float(y.mean())
        self._target_std = float(y.std()) or 1.0
        return (y - self._target_mean) / self._target_std

    def _unscale_y(self, y: np.ndarray) -> np.ndarray:
        return y * self._target_std + self._target_mean if self.scale_targets else y

    # ------------------------------------------------------------------
    # Hyperparameter optimization
    # ------------------------------------------------------------------

    def _nlml_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Negative log marginal likelihood + analytic gradient."""
        self._kernel.set_hyperparameters(theta)
        n = self._X_train.shape[0]
        K = self._kernel(self._X_train)
        K += (self.noise_variance + _JITTER) * np.eye(n)

        try:
            L, lower = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e18, np.zeros_like(theta)

        alpha = cho_solve((L, lower), self._y_train)

        nlml = (
            0.5 * float(self._y_train @ alpha)
            + float(np.sum(np.log(np.diag(L))))
            + 0.5 * n * _LOG_2PI
        )

        # ∂NLML/∂θᵢ = 0.5 tr[(αα ᵀ − K⁻¹) ∂K/∂θᵢ]
        K_inv = cho_solve((L, lower), np.eye(n))
        W = np.outer(alpha, alpha) - K_inv
        dK_list = self._kernel.gradient_wrt_hyperparams(self._X_train)
        grad = np.array([0.5 * float(np.einsum("ij,ji->", W, dK)) for dK in dK_list])

        return nlml, grad

    def _optimize_hyperparameters(self, rng: np.random.Generator) -> np.ndarray:
        from scipy.optimize import minimize

        n_params = self._kernel.n_hyperparameters
        theta0 = self._kernel.get_hyperparameters()

        # log-space bounds: amplitude [-4,4], length-scales/others [-3,3]
        bounds = [(-4.0, 4.0)] + [(-3.0, 3.0)] * (n_params - 1)

        best_nlml, best_theta = np.inf, theta0.copy()

        starts = [theta0] + [
            rng.uniform(-2.0, 2.0, size=n_params) for _ in range(self.n_restarts)
        ]

        for i, start in enumerate(starts):
            try:
                res = minimize(
                    self._nlml_and_grad,
                    start,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
                )
                if res.fun < best_nlml:
                    best_nlml, best_theta = res.fun, res.x.copy()
                    logger.debug(
                        "HP opt improved",
                        extra={
                            "restart": i,
                            "nlml": float(res.fun),
                            "success": res.success,
                        },
                    )
            except Exception:
                logger.debug("HP opt restart failed", extra={"restart": i})

        logger.info(
            "HP opt complete",
            extra={"best_nlml": float(best_nlml), "n_restarts": self.n_restarts},
        )
        return best_theta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> GaussianProcessRegressor:
        self.features_name = features_name
        rng = np.random.default_rng(self.seed)
        targets_flat = targets.ravel()

        self._X_train = self._scale_X(features, fit=True)
        self._y_train = self._scale_y(targets_flat)
        self._kernel = build_kernel(
            self.kernel_name, features.shape[1], degree=self.degree
        )

        best_theta = self._optimize_hyperparameters(rng)

        self._kernel.set_hyperparameters(best_theta)
        n = self._X_train.shape[0]
        K = self._kernel(self._X_train)
        K += (self.noise_variance + _JITTER) * np.eye(n)

        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, self._y_train)

        self._eval_metrics = {"train": _metrics(targets_flat, self.predict(features))}
        if eval_set is not None:
            X_e, y_e = eval_set
            self._eval_metrics["eval"] = _metrics(y_e.ravel(), self.predict(X_e))

        logger.info(
            "GP fit complete",
            extra={"metrics": self._eval_metrics, "kernel": self.kernel_name},
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        K_star = self._kernel(self._scale_X(features), self._X_train)
        return self._unscale_y(K_star @ self._alpha)

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) in original target scale."""
        self._check_fitted()
        X_test = self._scale_X(features)
        K_star = self._kernel(X_test, self._X_train)
        K_ss = self._kernel(X_test)

        mean_std = K_star @ self._alpha

        # v = L⁻¹ K*ᵀ  →  posterior var = diag(K**) - ‖v‖²
        L, lower = self._chol
        v = solve_triangular(
            L if lower else L.T,
            K_star.T,
            lower=lower,
            check_finite=False,
        )
        var_std = np.maximum(np.diag(K_ss) - np.sum(v**2, axis=0), 0.0)

        return self._unscale_y(mean_std), np.sqrt(var_std) * self._target_std

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._kernel is None or self._alpha is None or self._chol is None:
            raise ValueError("Model has not been fit yet. Call .fit() first.")

    def __str__(self) -> str:
        return f"GaussianProcess(kernel={self.kernel_name})"

    def get_fit_details(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "noise_variance": self.noise_variance,
            "scale_features": self.scale_features,
            "scale_targets": self.scale_targets,
            "hyperparameters": self._kernel.get_hyperparameters().tolist()
            if self._kernel
            else None,
            "feature_mean": self._feature_mean.tolist()
            if self._feature_mean is not None
            else None,
            "feature_std": self._feature_std.tolist()
            if self._feature_std is not None
            else None,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "eval_metrics": self._eval_metrics or None,
        }
