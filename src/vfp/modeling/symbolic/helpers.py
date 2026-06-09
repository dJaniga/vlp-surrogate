from __future__ import annotations

import logging
import os
import time
from copy import deepcopy

import numpy as np
from deap import creator, gp, tools
from scipy.optimize import minimize

from vfp.modeling.tuning_metrics import (
    evaluate_metric,
    AVAILABLE_METRICS,
    MAXIMIZE_METRICS,
)

logger = logging.getLogger(__name__)

# TODO: paramterize
_PENALTY_FITNESS = (1e18, 1e18)
MINIMIZE_METRICS = AVAILABLE_METRICS.difference(MAXIMIZE_METRICS)


def _compute_metric(preds: np.ndarray, targets: np.ndarray, metric: str) -> float:
    if metric not in MINIMIZE_METRICS:
        raise ValueError(
            f"Invalid metric: {metric} - only {MINIMIZE_METRICS} are supported for minimization"
        )
    return evaluate_metric(metric, preds, targets)


def build_seed_individuals(
    pset: gp.PrimitiveSet,
    n_features: int,
) -> list[gp.PrimitiveTree]:
    """Create hand-crafted seed individuals that use multiple features.

    These provide the GP with multi-variable starting structures that it
    can refine via crossover, mutation, and constant optimisation.
    """
    # Look up primitives and terminals from the pset - optimize dictionary building
    prim: dict[str, gp.Primitive] = {
        p.name: p for prims in pset.primitives.values() for p in prims
    }

    term: dict[str, gp.Terminal] = {
        t.name: t for terms in pset.terminals.values() for t in terms
    }

    # Cache frequently used primitives
    add_prim = prim["_add"]
    mul_prim = prim["_mul"]

    def _arg(i: int) -> gp.Terminal:
        return term[f"ARG{i}"]

    def _const(v: float) -> gp.Terminal:
        return make_constant_terminal(v)

    # Pre-allocate constant terminals to reuse
    const_zero = _const(0.0)

    seeds: list[list[gp.Primitive | gp.Terminal]] = []

    # 1. Linear: c0 + c1*ARG0 + c2*ARG1  (all pairwise combos of 2 features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # _add(_mul(c1, ARGi), _mul(c2, ARGj))
            tokens = [
                add_prim,
                mul_prim,
                _const(1.0),
                _arg(i),
                mul_prim,
                _const(1.0),
                _arg(j),
            ]
            seeds.append(tokens)

    # 2. Linear with all features: c0 + c1*ARG0 + c2*ARG1 + c3*ARG2 + c4*ARG3
    if n_features >= 2:
        # Build: _add(_add(...), _mul(c, ARGn))
        # Start with _mul(c, ARG0)
        tokens = [mul_prim, _const(1.0), _arg(0)]
        for i in range(1, n_features):
            tokens = [add_prim] + tokens + [mul_prim, _const(1.0), _arg(i)]
        # Add a constant offset: _add(c0, <above>)
        tokens = [add_prim, const_zero] + tokens
        seeds.append(tokens)

    # 3. Quadratic in main feature + linear in others:
    #    c0 + c1*ARG0 + c2*ARG0^2 + c3*ARG1
    if "_square" in prim:
        square_prim = prim["_square"]
        for main in range(min(n_features, 2)):
            other = 1 - main
            tokens = [
                add_prim,
                add_prim,
                mul_prim,
                _const(1.0),
                _arg(main),
                mul_prim,
                _const(1.0),
                square_prim,
                _arg(main),
                mul_prim,
                _const(1.0),
                _arg(other),
            ]
            seeds.append(tokens)

    # 4. Product interaction: c * ARG0 * ARG1
    if n_features >= 2:
        tokens = [
            mul_prim,
            _const(1.0),
            mul_prim,
            _arg(0),
            _arg(1),
        ]
        seeds.append(tokens)

    # 5. Ratio: ARG0 / ARG1
    if n_features >= 2:
        tokens = [
            prim["_protected_div"],
            _arg(0),
            _arg(1),
        ]
        seeds.append(tokens)

    # 6. Affine pairwise: c0 + c1*ARGi + c2*ARGj  (with explicit intercept)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            tokens = [
                add_prim,
                add_prim,
                _const(0.0),  # intercept c0
                mul_prim,
                _const(1.0),
                _arg(i),
                mul_prim,
                _const(1.0),
                _arg(j),
            ]
            seeds.append(tokens)

    # 7. Scaled difference: c * (ARGi - ARGj)
    if "_sub" in prim:
        sub_prim = prim["_sub"]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                tokens = [
                    mul_prim,
                    _const(1.0),
                    sub_prim,
                    _arg(i),
                    _arg(j),
                ]
                seeds.append(tokens)
    # 8. Leave-one-out linear combination (n_features >= 3)
    if n_features >= 3:
        for skip in range(n_features):
            active = [k for k in range(n_features) if k != skip]
            tokens = [mul_prim, _const(1.0), _arg(active[0])]
            for k in active[1:]:
                tokens = [add_prim] + tokens + [mul_prim, _const(1.0), _arg(k)]
            tokens = [add_prim, const_zero] + tokens
            seeds.append(tokens)

    # 9. Linear + all pairwise interactions:
    #    c0 + c1*ARG0 + ... + cN*ARG(N-1) + c_ij*ARGi*ARGj  (for all i < j)
    if n_features >= 2 and n_features <= 6:
        # Start with main effects
        terms: list[list] = [
            [mul_prim, _const(1.0), _arg(i)] for i in range(n_features)
        ]
        # Add interaction terms: c_ij * ARGi * ARGj
        for i in range(n_features):
            for j in range(i + 1, n_features):
                terms.append([mul_prim, _const(1.0), mul_prim, _arg(i), _arg(j)])
        # Chain all terms with _add: _add(term0, _add(term1, ...))
        tokens = terms[0]
        for term in terms[1:]:
            tokens = [add_prim] + tokens + term
        # Prepend intercept
        tokens = [add_prim, const_zero] + tokens
        seeds.append(tokens)

    individuals: list[gp.PrimitiveTree] = []
    for token_list in seeds:
        try:
            ind = creator.SymbolicIndividual(token_list)  # type: ignore[attr-defined]
            individuals.append(ind)
        except Exception:
            continue

    return individuals


def vectorised_evaluate(func: object, features: np.ndarray) -> np.ndarray:
    try:
        n_cols = features.shape[1]
        columns = [features[:, i] for i in range(n_cols)]
        result = func(*columns)
        if result is None:
            return _safe_evaluate_rows(func, features)
        arr = np.asarray(result, dtype=np.float64)
        if arr.shape != (features.shape[0],):
            arr = np.full(features.shape[0], float(arr), dtype=np.float64)
        # np.all on isfinite short-circuits at C level
        if np.all(np.isfinite(arr)):
            return arr
        return _safe_evaluate_rows(func, features)
    except Exception:
        return _safe_evaluate_rows(func, features)


def _safe_evaluate_rows(func: object, features: np.ndarray) -> np.ndarray:
    """Row-by-row fallback with NaN for failures.

    If the whole batch takes more than ``EVAL_ROW_TIMEOUT`` seconds the
    remaining rows are filled with NaN so the caller sees a (penalised)
    finite result rather than hanging indefinitely.
    """
    EVAL_ROW_TIMEOUT: float = float(os.environ.get("VLP_EVAL_ROW_TIMEOUT", "5.0"))
    results = np.empty(features.shape[0], dtype=float)
    deadline = time.monotonic() + EVAL_ROW_TIMEOUT
    for i, row in enumerate(features):
        if time.monotonic() > deadline:
            results[i:] = np.nan
            logger.warning(
                "Row-level evaluation timed out after %.1fs; filling %d rows with NaN",
                EVAL_ROW_TIMEOUT,
                features.shape[0] - i,
            )
            break
        try:
            value = func(*row)  # type: ignore[operator]
            if value is None:
                results[i] = np.nan
            else:
                results[i] = float(value)
        except (
            TypeError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
        ):
            results[i] = np.nan
    return results


def make_constant_terminal(value: float) -> gp.Terminal:
    """Create a constant terminal with ``object`` return type (matches the untyped pset)."""
    # The value is explicitly cast to a Python float so that repr(value)
    # produces a plain literal like 1.0 rather than np.float64(1.0).
    return gp.Terminal(float(value), False, object)


_COMPILE_CACHE: dict[str, object] = {}
_COMPILE_CACHE_MAX = 20_000


def _compile_cached(individual: gp.PrimitiveTree, pset: gp.PrimitiveSet) -> object:
    key = str(individual)
    func = _COMPILE_CACHE.get(key)
    if func is None:
        func = gp.compile(individual, pset)
        if len(_COMPILE_CACHE) < _COMPILE_CACHE_MAX:
            _COMPILE_CACHE[key] = func
    return func


def optimize_constants(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    timeout: float | None = None,
) -> None:
    """Optimise numeric constants in *individual* in-place.

    Parameters
    ----------
    timeout:
        Wall-clock budget in seconds.  Defaults to the env var
        ``VLP_CONST_OPT_TIMEOUT`` (float, default ``2.0``).  The Nelder-Mead
        callback raises ``StopIteration`` when the budget is exhausted, so the
        best result found *so far* is still applied.
    """
    if timeout is None:
        timeout = float(os.environ.get("VLP_CONST_OPT_TIMEOUT", "2.0"))

    indices = [
        idx
        for idx, node in enumerate(individual)
        if isinstance(node, gp.Terminal) and isinstance(node.value, (int, float))
    ]
    if not indices:
        return

    initial = np.array([float(individual[idx].value) for idx in indices], dtype=float)
    inv_n = 1.0 / len(targets)
    diff_buf = np.empty(len(targets), dtype=np.float64)  # reusable buffer

    def objective(constants: np.ndarray) -> float:
        for idx, value in zip(indices, constants, strict=False):
            individual[idx] = make_constant_terminal(float(value))
        try:
            func = _compile_cached(individual, pset)
            preds = vectorised_evaluate(func, features)
        except Exception:
            return 1e18
        if not np.isfinite(preds).all():
            return 1e18
        np.subtract(preds, targets, out=diff_buf)
        return float(np.dot(diff_buf, diff_buf) * inv_n)

    initial_fitness = objective(initial)

    # Adaptive iteration budget: small trees converge fast.
    # Capped lower than before to prevent runaway optimisation on large trees.
    maxiter = min(50, 10 + 5 * len(indices))

    deadline = time.monotonic() + timeout

    def _timeout_callback(intermediate_result) -> None:  # noqa: ANN001
        if time.monotonic() > deadline:
            raise StopIteration("optimize_constants timeout")

    try:
        result = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            callback=_timeout_callback,
            options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-7},
        )
        best = result.x if result.fun < initial_fitness else initial
    except StopIteration:
        logger.debug(
            "optimize_constants interrupted by timeout (%.1fs budget)", timeout
        )
        # Keep whatever the objective() loop already wrote into individual.
        return

    for idx, value in zip(indices, best, strict=False):
        individual[idx] = make_constant_terminal(float(value))


def evaluate_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    parsimony_coefficient: float,
    max_tree_height: int,
) -> tuple[float, float]:
    """Evaluate fitness as (penalised_error, tree_complexity).

    error increase tolerated for a fully-bloated tree.
    E.g. 0.05 = allow up to 5% worse error in exchange for maximum tree size.
    """
    fit_metric = os.environ.get("VLP_FIT_METRIC")
    try:
        func = _compile_cached(individual, pset)
        preds = vectorised_evaluate(func, features)
        if not np.all(np.isfinite(preds)):
            return _PENALTY_FITNESS
        error = _compute_metric(preds, targets, fit_metric)
        if not np.isfinite(error):
            return _PENALTY_FITNESS
    except Exception:
        return _PENALTY_FITNESS

    complexity = float(individual.height) / max_tree_height
    penalized_error = error + parsimony_coefficient * complexity

    return penalized_error, complexity


def migrate(
    islands: list[list[gp.PrimitiveTree]],
    migration_size: int,
    rng: np.random.Generator,
) -> None:
    """Ring-topology migration: each island sends its best to the next island.

    The ``migration_size`` best individuals from island *i* replace the worst
    ``migration_size`` individuals in island *(i+1) % n*.
    """
    n = len(islands)
    if n < 2:
        return

    # Pre-compute emigrants for all islands
    emigrants: list[list[gp.PrimitiveTree]] = []
    for island in islands:
        best = tools.selBest(island, k=min(migration_size, len(island)))
        emigrants.append([deepcopy(ind) for ind in best])

    # Perform migration more efficiently
    for i in range(n):
        dest = (i + 1) % n
        dest_island = islands[dest]
        k = min(migration_size, len(dest_island))
        worst = tools.selWorst(dest_island, k=k)

        # Use set for O(1) lookup instead of O(n) list.remove()
        worst_set = set(id(w) for w in worst)
        islands[dest] = [ind for ind in dest_island if id(ind) not in worst_set]
        islands[dest].extend(emigrants[i])

    logger.debug(
        "Migration complete",
        extra={"islands": n, "migrants_per_island": migration_size},
    )