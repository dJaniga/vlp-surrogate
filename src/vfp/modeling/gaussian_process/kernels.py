"""GP kernel library with analytic gradients for L-BFGS-B hyperparameter optimization.

All kernels expose:
  - __call__(X, Y=None)               → kernel matrix
  - gradient_wrt_hyperparams(X)       → list[np.ndarray], one (n,n) matrix per log-param
  - get_hyperparameters() / set_hyperparameters(theta)
  - n_hyperparameters (property)

All hyperparameters are stored and optimized in **log-space** so they remain
positive without explicit constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — what every kernel must expose
# ---------------------------------------------------------------------------


@runtime_checkable
class Kernel(Protocol):
    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray: ...
    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]: ...
    def get_hyperparameters(self) -> np.ndarray: ...
    def set_hyperparameters(self, theta: np.ndarray) -> None: ...
    @property
    def n_hyperparameters(self) -> int: ...


# ---------------------------------------------------------------------------
# RBF  (isotropic)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RBFKernel:
    """k(x,x') = σ² exp(-‖x-x'‖² / (2 l²))

    Log-params: [log σ², log l]
    """

    signal_variance: float = 1.0
    length_scale: float = 1.0

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        r2 = cdist(X, Y, metric="sqeuclidean")
        return self.signal_variance * np.exp(-0.5 * r2 / self.length_scale**2)

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        K = self(X)
        r2 = cdist(X, X, metric="sqeuclidean")
        # ∂K/∂(log σ²) = K
        # ∂K/∂(log l)  = K * r²/l²   (chain rule through log-space)
        return [K, K * r2 / self.length_scale**2]

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.array([self.signal_variance, self.length_scale]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scale = float(np.exp(theta[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# ARD RBF  (per-feature length scales)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ARDKernel:
    """k(x,x') = σ² exp(-0.5 Σ_d ((x_d-x'_d)/l_d)²)

    Log-params: [log σ², log l₀, …, log l_{d-1}]
    """

    signal_variance: float = 1.0
    length_scales: np.ndarray = field(default_factory=lambda: np.ones(1))

    def _scaled_sq_dists(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (r², per-dim sq-dists) both shape (nX, nY) and (nX, nY, d)."""
        diff = (X[:, None, :] - Y[None, :, :]) / self.length_scales  # (nX,nY,d)
        sq = diff**2
        return sq.sum(axis=-1), sq

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        r2, _ = self._scaled_sq_dists(X, Y)
        return self.signal_variance * np.exp(-0.5 * r2)

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        r2, sq_per_dim = self._scaled_sq_dists(X, X)
        K = self.signal_variance * np.exp(-0.5 * r2)
        # ∂K/∂(log σ²) = K
        # ∂K/∂(log lₖ) = K * (xₖ-x'ₖ)²/lₖ²   per dimension
        grads = [K]
        for k in range(len(self.length_scales)):
            grads.append(K * sq_per_dim[:, :, k])
        return grads

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.concatenate([[self.signal_variance], self.length_scales]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scales = np.exp(theta[1:])

    @property
    def n_hyperparameters(self) -> int:
        return 1 + len(self.length_scales)


# ---------------------------------------------------------------------------
# Matern 5/2  (isotropic)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Matern52Kernel:
    """k(x,x') = σ² (1 + √5 r + 5r²/3) exp(-√5 r),  r = ‖x-x'‖/l

    Log-params: [log σ², log l]
    """

    signal_variance: float = 1.0
    length_scale: float = 1.0

    def _r_and_K(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = cdist(X, Y, metric="euclidean") / self.length_scale
        sqrt5_r = np.sqrt(5.0) * r
        K = self.signal_variance * (1.0 + sqrt5_r + 5.0 / 3.0 * r**2) * np.exp(-sqrt5_r)
        return r, K

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        return self._r_and_K(X, X if Y is None else Y)[1]

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        r, K = self._r_and_K(X, X)
        # dK/dr = -5/3 σ² (1 + √5 r) r exp(-√5 r)
        dK_dr = (
            -5.0
            / 3.0
            * self.signal_variance
            * (1.0 + np.sqrt(5.0) * r)
            * r
            * np.exp(-np.sqrt(5.0) * r)
        )
        # dr/d(log l) = -r  (log-space chain rule)
        return [K, dK_dr * (-r)]

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.array([self.signal_variance, self.length_scale]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scale = float(np.exp(theta[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Matern 5/2 ARD  (per-feature length scales) — new, recommended for 4-feature use
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Matern52ARDKernel:
    """k(x,x') = σ² (1 + √5 r + 5r²/3) exp(-√5 r),
    r = sqrt(Σ_d ((x_d-x'_d)/l_d)²)

    Per-feature length scales let the model down-weight irrelevant features.
    Log-params: [log σ², log l₀, …, log l_{d-1}]
    """

    signal_variance: float = 1.0
    length_scales: np.ndarray = field(default_factory=lambda: np.ones(1))

    def _r2_diff(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        diff = (X[:, None, :] - Y[None, :, :]) / self.length_scales  # (nX,nY,d)
        sq = diff**2
        r2 = sq.sum(axis=-1)
        return r2, sq, diff

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        r2, _, _ = self._r2_diff(X, Y)
        r = np.sqrt(np.maximum(r2, 0.0))
        sqrt5_r = np.sqrt(5.0) * r
        return (
            self.signal_variance * (1.0 + sqrt5_r + 5.0 / 3.0 * r2) * np.exp(-sqrt5_r)
        )

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        r2, sq_per_dim, _ = self._r2_diff(X, X)
        r = np.sqrt(np.maximum(r2, 0.0))
        sqrt5_r = np.sqrt(5.0) * r
        exp_t = np.exp(-sqrt5_r)
        K = self.signal_variance * (1.0 + sqrt5_r + 5.0 / 3.0 * r2) * exp_t

        # dK/dr = -5/3 σ² (1 + √5 r) r exp(-√5 r)
        safe_r = np.where(r > 1e-12, r, 1.0)
        dK_dr = -5.0 / 3.0 * self.signal_variance * (1.0 + sqrt5_r) * r * exp_t

        grads = [K]  # ∂K/∂(log σ²) = K
        for k in range(len(self.length_scales)):
            # dr/d(log lₖ) = -sq_per_dim[:,:,k] / r
            dr_dlogl = np.where(r > 1e-12, -sq_per_dim[:, :, k] / safe_r, 0.0)
            grads.append(dK_dr * dr_dlogl)
        return grads

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.concatenate([[self.signal_variance], self.length_scales]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scales = np.exp(theta[1:])

    @property
    def n_hyperparameters(self) -> int:
        return 1 + len(self.length_scales)


# ---------------------------------------------------------------------------
# Linear  (new — captures linear trends directly)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LinearKernel:
    """k(x,x') = σ_b² + σ_v² (x · x')

    σ_b²: bias variance (vertical shift)
    σ_v²: slope variance (scales the inner product)

    Log-params: [log σ_b², log σ_v²]
    """

    bias_variance: float = 1.0
    slope_variance: float = 1.0

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        return self.bias_variance + self.slope_variance * (X @ Y.T)

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        n = X.shape[0]
        dot = X @ X.T
        # ∂K/∂(log σ_b²) = σ_b² · 1   (log-space chain rule)
        # ∂K/∂(log σ_v²) = σ_v² · XXT
        return [
            self.bias_variance * np.ones((n, n)),
            self.slope_variance * dot,
        ]

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.array([self.bias_variance, self.slope_variance]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.bias_variance = float(np.exp(theta[0]))
        self.slope_variance = float(np.exp(theta[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PolynomialKernel:
    """k(x,x') = σ² (x·x' + c)^degree

    Log-params: [log σ², log c]
    degree is a fixed integer (not optimized).
    """

    signal_variance: float = 1.0
    coef0: float = 1.0
    degree: int = 3

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        return self.signal_variance * (X @ Y.T + self.coef0) ** self.degree

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        inner = X @ X.T + self.coef0
        base = inner**self.degree
        # ∂K/∂(log σ²) = σ² · base
        # ∂K/∂(log c)  = σ² · degree · inner^(degree-1) · c
        return [
            self.signal_variance * base,
            self.signal_variance
            * self.degree
            * inner ** (self.degree - 1)
            * self.coef0,
        ]

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.array([self.signal_variance, self.coef0]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.coef0 = float(np.exp(theta[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Rational Quadratic
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RationalQuadraticKernel:
    """k(x,x') = σ² (1 + ‖x-x'‖² / (2α l²))^(-α)

    Log-params: [log σ², log l, log α]
    """

    signal_variance: float = 1.0
    length_scale: float = 1.0
    alpha: float = 1.0

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        r2 = cdist(X, Y, metric="sqeuclidean")
        return self.signal_variance * (
            1.0 + r2 / (2.0 * self.alpha * self.length_scale**2)
        ) ** (-self.alpha)

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        r2 = cdist(X, X, metric="sqeuclidean")
        denom = 2.0 * self.alpha * self.length_scale**2
        base = 1.0 + r2 / denom  # (1 + r²/2αl²)
        K = self.signal_variance * base ** (-self.alpha)

        # ∂K/∂(log σ²) = K
        dK_dsig = K

        # ∂K/∂(log l) = K · α · r² / (α l² + r²/2)  * 1  (log-space)
        #             = K · r² / (l² + r²/(2α))  ... simplify via base
        dK_dl = K * self.alpha * (r2 / denom) / base * (-1.0) * (-2.0)
        # cleaner: ∂K/∂(log l) = σ² (-α) base^(-α-1) · (-r²/αl²) · l
        #                      = K · r² / (l² base)  ... but let's be explicit:
        dK_dl = K * (r2 / (self.length_scale**2 * base))

        # ∂K/∂(log α) = K · [-log(base) + r²/(2αl² base)] · α  (log-space chain rule)
        dK_da = K * self.alpha * (r2 / (denom * base) - np.log(base))

        return [dK_dsig, dK_dl, dK_da]

    def get_hyperparameters(self) -> np.ndarray:
        return np.log(np.array([self.signal_variance, self.length_scale, self.alpha]))

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.signal_variance = float(np.exp(theta[0]))
        self.length_scale = float(np.exp(theta[1]))
        self.alpha = float(np.exp(theta[2]))

    @property
    def n_hyperparameters(self) -> int:
        return 3


# ---------------------------------------------------------------------------
# Composite (sum of two kernels)  — new
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompositeKernel:
    """Sum of two kernels: k(x,x') = k₁(x,x') + k₂(x,x')

    Hyperparameters are the concatenation of both kernels' hyperparameters.
    Use CompositeKernel(Matern52ARDKernel(...), LinearKernel()) for the
    recommended 4-feature regression setup.
    """

    k1: object  # any Kernel
    k2: object  # any Kernel

    @property
    def _n1(self) -> int:
        return self.k1.n_hyperparameters

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        return self.k1(X, Y) + self.k2(X, Y)

    def gradient_wrt_hyperparams(self, X: np.ndarray) -> list[np.ndarray]:
        return self.k1.gradient_wrt_hyperparams(X) + self.k2.gradient_wrt_hyperparams(X)

    def get_hyperparameters(self) -> np.ndarray:
        return np.concatenate(
            [
                self.k1.get_hyperparameters(),
                self.k2.get_hyperparameters(),
            ]
        )

    def set_hyperparameters(self, theta: np.ndarray) -> None:
        self.k1.set_hyperparameters(theta[: self._n1])
        self.k2.set_hyperparameters(theta[self._n1 :])

    @property
    def n_hyperparameters(self) -> int:
        return self.k1.n_hyperparameters + self.k2.n_hyperparameters


# ---------------------------------------------------------------------------
# Type alias & factory
# ---------------------------------------------------------------------------

type KernelType = (
    RBFKernel
    | ARDKernel
    | Matern52Kernel
    | Matern52ARDKernel
    | LinearKernel
    | PolynomialKernel
    | RationalQuadraticKernel
    | CompositeKernel
)


def build_kernel(name: str, n_features: int, *, degree: int = 3) -> KernelType:
    """Instantiate a kernel by name.

    Names
    -----
    'rbf'                isotropic RBF
    'ard'                ARD RBF (per-feature length scales)
    'matern52'           isotropic Matern 5/2
    'matern52_ard'       ARD Matern 5/2  ← recommended base for regression
    'linear'             linear (bias + slope)
    'composite'          Matern52ARD + Linear  ← recommended for linear trends
    'polynomial'         polynomial degree `degree`
    'rational_quadratic' rational quadratic
    """
    ls = np.ones(n_features)

    if name == "rbf":
        return RBFKernel()
    if name == "ard":
        return ARDKernel(length_scales=ls.copy())
    if name == "matern52":
        return Matern52Kernel()
    if name == "matern52_ard":
        return Matern52ARDKernel(length_scales=ls.copy())
    if name == "linear":
        return LinearKernel()
    if name == "composite":
        return CompositeKernel(
            k1=Matern52ARDKernel(length_scales=ls.copy()),
            k2=LinearKernel(),
        )
    if name == "polynomial":
        return PolynomialKernel(degree=degree)
    if name == "rational_quadratic":
        return RationalQuadraticKernel()

    raise ValueError(
        f"Unknown kernel: {name!r}. Choose from "
        "'rbf', 'ard', 'matern52', 'matern52_ard', 'linear', "
        "'composite', 'polynomial', 'rational_quadratic'."
    )
