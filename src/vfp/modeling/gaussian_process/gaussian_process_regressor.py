from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from vfp.modeling.base import VFPModel
from vfp.modeling.gaussian_process.kernels import Kernel, build_kernel

logger = logging.getLogger(__name__)

_JITTER = 1e-8


@dataclass(slots=True)
class GaussianProcessRegressor(VFPModel):
    """Gaussian Process regressor for VFP surrogate modeling.

    Fits a GP with noise-free observations (simulation data) and optimizes
    kernel hyperparameters by maximizing the log marginal likelihood.
    """

    kernel_name: str = "rbf"
    noise_variance: float = 1e-6
    n_restarts: int = 5
    seed: int | None = None
    _kernel: Kernel | None = None
    _X_train: np.ndarray | None = None
    _y_train: np.ndarray | None = None
    _alpha: np.ndarray | None = None
    _L: np.ndarray | None = None
    _feature_mean: np.ndarray | None = None
    _feature_std: np.ndarray | None = None
    _target_mean: float = 0.0
    _target_std: float = 1.0

    def _standardize_features(
        self, features: np.ndarray, *, fit: bool = False
    ) -> np.ndarray:
        if fit:
            self._feature_mean = np.asarray(features.mean(axis=0))
            std = np.asarray(features.std(axis=0))
            std[std < 1e-12] = 1.0
            self._feature_std = std
        assert self._feature_mean is not None and self._feature_std is not None
        return (features - self._feature_mean) / self._feature_std

    def _standardize_targets(self, targets: np.ndarray) -> np.ndarray:
        self._target_mean = float(targets.mean())
        self._target_std = float(targets.std())
        if self._target_std < 1e-12:
            self._target_std = 1.0
        return (targets - self._target_mean) / self._target_std

    def _unstandardize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        return predictions * self._target_std + self._target_mean

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> GaussianProcessRegressor:
        self.features_name = features_name
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = features.shape

        targets_flat = targets.ravel()

        logger.debug(
            "Fitting Gaussian Process",
            extra={
                "samples": n_samples,
                "features": n_features,
                "kernel": self.kernel_name,
            },
        )

        self._X_train = self._standardize_features(features, fit=True)
        self._y_train = self._standardize_targets(targets_flat)

        self._kernel = build_kernel(self.kernel_name, n_features)

        best_theta = self._optimize_hyperparameters(rng)

        self._kernel.set_hyperparameters(best_theta)
        K = self._kernel(self._X_train)
        K += (self.noise_variance + _JITTER) * np.eye(n_samples)

        L, lower = cho_factor(K, lower=True)
        self._L = L
        self._alpha = cho_solve((L, lower), self._y_train)

        logger.debug(
            "Gaussian Process fit complete",
            extra={
                "kernel": self.kernel_name,
                "noise_variance": self.noise_variance,
                "n_restarts": self.n_restarts,
            },
        )
        return self

    def _optimize_hyperparameters(self, rng: np.random.Generator) -> np.ndarray:
        """Optimize kernel hyperparameters by maximizing log marginal likelihood."""
        assert self._kernel is not None
        assert self._X_train is not None
        assert self._y_train is not None

        n_params = self._kernel.n_hyperparameters
        initial_theta = self._kernel.get_hyperparameters()

        best_nlml = np.inf
        best_theta = initial_theta.copy()

        starting_points = [initial_theta]
        for _ in range(self.n_restarts):
            starting_points.append(rng.uniform(-2.0, 2.0, size=n_params))

        for i, theta0 in enumerate(starting_points):
            try:
                result = minimize(
                    self._negative_log_marginal_likelihood,
                    theta0,
                    method="L-BFGS-B",
                    options={"maxiter": 200},
                )
                if result.fun < best_nlml:
                    best_nlml = result.fun
                    best_theta = result.x
                    logger.debug(
                        "Hyperparameter optimization improved",
                        extra={
                            "restart": i,
                            "nlml": float(result.fun),
                            "success": result.success,
                        },
                    )
            except Exception:
                logger.debug(
                    "Hyperparameter optimization restart failed",
                    extra={"restart": i},
                )
                continue

        logger.info(
            "Hyperparameter optimization complete",
            extra={
                "best_nlml": float(best_nlml),
                "n_restarts": self.n_restarts,
            },
        )
        return best_theta

    def _negative_log_marginal_likelihood(self, theta: np.ndarray) -> float:
        """Compute the negative log marginal likelihood for hyperparameter optimization."""
        assert self._kernel is not None
        assert self._X_train is not None
        assert self._y_train is not None

        self._kernel.set_hyperparameters(theta)
        n = self._X_train.shape[0]
        K = self._kernel(self._X_train)
        K += (self.noise_variance + _JITTER) * np.eye(n)

        try:
            L, lower = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e18

        alpha = cho_solve((L, lower), self._y_train)

        # log p(y|X, theta) = -0.5 * y^T * alpha - sum(log(diag(L))) - n/2 * log(2*pi)
        data_fit = -0.5 * float(self._y_train @ alpha)
        complexity = -float(np.sum(np.log(np.diag(L))))
        constant = -0.5 * n * np.log(2.0 * np.pi)

        log_ml = data_fit + complexity + constant
        return -log_ml

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            self._kernel is None
            or self._X_train is None
            or self._alpha is None
            or self._L is None
        ):
            raise ValueError("Model has not been fit yet.")

        X_test = self._standardize_features(features)
        K_star = self._kernel(X_test, self._X_train)

        predictions_std = K_star @ self._alpha
        predictions = self._unstandardize_predictions(predictions_std)

        logger.debug(
            "Gaussian Process prediction complete",
            extra={"samples": int(features.shape[0])},
        )
        return predictions

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation.

        Returns the predictive mean and standard deviation in the original
        (unstandardized) target scale.
        """
        if (
            self._kernel is None
            or self._X_train is None
            or self._alpha is None
            or self._L is None
        ):
            raise ValueError("Model has not been fit yet.")

        X_test = self._standardize_features(features)
        K_star = self._kernel(X_test, self._X_train)
        K_ss = self._kernel(X_test)

        mean_std = K_star @ self._alpha

        v = cho_solve((self._L, True), K_star.T)
        var_std = np.diag(K_ss) - np.sum(K_star * v.T, axis=1)
        var_std = np.maximum(var_std, 0.0)

        mean = self._unstandardize_predictions(mean_std)
        std = np.sqrt(var_std) * self._target_std

        logger.debug(
            "Gaussian Process prediction with uncertainty complete",
            extra={"samples": int(features.shape[0])},
        )
        return mean, std

    def __str__(self) -> str:
        return f"GaussianProcess_{self.kernel_name}"

    def get_fit_details(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "noise_variance": self.noise_variance,
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
        }
