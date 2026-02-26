from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RBFKernel:
    """Radial Basis Function (squared-exponential) kernel.

    k(x, x') = signal_variance * exp(-0.5 * ||x - x'||^2 / length_scale^2)

    Hyperparameters (log-space): [log(signal_variance), log(length_scale)]
    """

    signal_variance: float = 1.0
    length_scale: float = 1.0

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        dists_sq = cdist(X, Y, metric="sqeuclidean")
        return self.signal_variance * np.exp(-0.5 * dists_sq / (self.length_scale**2))

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.array([self.signal_variance, self.length_scale]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scale = float(np.exp(theta[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


@dataclass(slots=True)
class ARDKernel:
    """Automatic Relevance Determination (ARD) RBF kernel.

    Uses a separate length scale per input dimension, allowing the model to
    learn which features are most relevant.

    k(x, x') = signal_variance * exp(-0.5 * sum_d ((x_d - x'_d) / l_d)^2)

    Hyperparameters (log-space): [log(signal_variance), log(l_0), ..., log(l_{d-1})]
    """

    signal_variance: float = 1.0
    length_scales: np.ndarray | None = None

    def _ensure_length_scales(self, n_features: int) -> np.ndarray:
        if self.length_scales is None or self.length_scales.shape[0] != n_features:
            self.length_scales = np.ones(n_features)
        return self.length_scales

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        ls = self._ensure_length_scales(X.shape[1])
        X_scaled = X / ls
        Y_scaled = Y / ls
        dists_sq = cdist(X_scaled, Y_scaled, metric="sqeuclidean")
        return self.signal_variance * np.exp(-0.5 * dists_sq)

    def get_hyperparameters(self) -> np.ndarray:
        ls = self._ensure_length_scales(1)  # fallback; should be set by fit
        return np.log(np.concatenate([[self.signal_variance], ls]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scales = np.exp(theta[1:])

    @property
    def n_hyperparameters(self) -> int:
        n_ls = self.length_scales.shape[0] if self.length_scales is not None else 1
        return 1 + n_ls


@dataclass(slots=True)
class Matern52Kernel:
    """Matern 5/2 kernel.

    k(x, x') = signal_variance * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    where r = ||x - x'|| / length_scale

    Hyperparameters (log-space): [log(signal_variance), log(length_scale)]
    """

    signal_variance: float = 1.0
    length_scale: float = 1.0

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        dists = cdist(X, Y, metric="euclidean")
        r = dists / self.length_scale
        sqrt5_r = np.sqrt(5.0) * r
        return (
            self.signal_variance * (1.0 + sqrt5_r + 5.0 / 3.0 * r**2) * np.exp(-sqrt5_r)
        )

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.array([self.signal_variance, self.length_scale]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scale = float(np.exp(theta[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


type Kernel = RBFKernel | ARDKernel | Matern52Kernel


def build_kernel(name: str, n_features: int) -> Kernel:
    """Build a kernel by name.

    Supported names: 'rbf', 'ard', 'matern52'.
    """
    if name == "rbf":
        return RBFKernel()
    if name == "ard":
        k = ARDKernel()
        k._ensure_length_scales(n_features)
        return k
    if name == "matern52":
        return Matern52Kernel()
    raise ValueError(f"Unknown kernel: {name!r}. Choose from 'rbf', 'ard', 'matern52'.")
