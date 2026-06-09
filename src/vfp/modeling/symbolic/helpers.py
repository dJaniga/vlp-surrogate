from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
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

_PENALTY_FITNESS = (1e18, 1e18)
MINIMIZE_METRICS = AVAILABLE_METRICS.difference(MAXIMIZE_METRICS)

# ---- perf #4: read env vars once at module load, not on every call ----------
_FIT_METRIC: str | None = os.environ.get("VLP_FIT_METRIC")
_CONST_OPT_TIMEOUT: float = float(os.environ.get("VLP_CONST_OPT_TIMEOUT", "2.0"))
_EVAL_ROW_TIMEOUT: float = float(os.environ.get("VLP_EVAL_ROW_TIMEOUT", "5.0"))
# perf #6: resolve force-simplicity flag once
_FORCE_SIMPLICITY: bool = os.environ.get("VLP_FORCE_SYMBOLIC_SIMPLICITY", "").lower() == "true"

# ---- perf #9: LRU fitness cache (OrderedDict-based, evicts oldest half) -----
_FITNESS_CACHE_MAX = 20_000
_COMPILE_CACHE: OrderedDict[str, object] = OrderedDict()
_COMPILE_CACHE_MAX = 20_000


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
    """Create hand-crafted seed individuals that use multiple features."""
    prim: dict[str, gp.Primitive] = {
        p.name: p for prims in pset.primitives.values() for p in prims
    }
    term: dict[str, gp.Terminal] = {
        t.name: t for terms in pset.terminals.values() for t in terms
    }

    add_prim = prim["_add"]
    mul_prim = prim["_mul"]

    def _arg(i: int) -> gp.Terminal:
        return term[f"ARG{i}"]

    def _const(v: float) -> gp.Terminal:
        return make_constant_terminal(v)

    const_zero = _const(0.0)
    seeds: list[list[gp.Primitive | gp.Terminal]] = []

    # 1. Linear pairwise: c1*ARGi + c2*ARGj
    for i in range(n_features):
        for j in range(i + 1, n_features):
            seeds.append([add_prim, mul_prim, _const(1.0), _arg(i), mul_prim, _const(1.0), _arg(j)])

    # 2. Linear with all features
    if n_features >= 2:
        tokens = [mul_prim, _const(1.0), _arg(0)]
        for i in range(1, n_features):
            tokens = [add_prim] + tokens + [mul_prim, _const(1.0), _arg(i)]
        seeds.append([add_prim, const_zero] + tokens)

    # 3. Quadratic in main feature + linear in other
    if "_square" in prim:
        square_prim = prim["_square"]
        for main in range(min(n_features, 2)):
            other = 1 - main
            seeds.append([
                add_prim, add_prim,
                mul_prim, _const(1.0), _arg(main),
                mul_prim, _const(1.0), square_prim, _arg(main),
                mul_prim, _const(1.0), _arg(other),
            ])

    # 4. Product interaction: c * ARG0 * ARG1
    if n_features >= 2:
        seeds.append([mul_prim, _const(1.0), mul_prim, _arg(0), _arg(1)])

    # 5. Ratio: ARG0 / ARG1
    if n_features >= 2:
        seeds.append([prim["_protected_div"], _arg(0), _arg(1)])

    # 6. Affine pairwise with explicit intercept
    for i in range(n_features):
        for j in range(i + 1, n_features):
            seeds.append([
                add_prim, add_prim, _const(0.0),
                mul_prim, _const(1.0), _arg(i),
                mul_prim, _const(1.0), _arg(j),
            ])

    # 7. Scaled difference: c * (ARGi - ARGj)
    if "_sub" in prim:
        sub_prim = prim["_sub"]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                seeds.append([mul_prim, _const(1.0), sub_prim, _arg(i), _arg(j)])

    # 8. Leave-one-out linear combination (n_features >= 3)
    if n_features >= 3:
        for skip in range(n_features):
            active = [k for k in range(n_features) if k != skip]
            tokens = [mul_prim, _const(1.0), _arg(active[0])]
            for k in active[1:]:
                tokens = [add_prim] + tokens + [mul_prim, _const(1.0), _arg(k)]
            seeds.append([add_prim, const_zero] + tokens)

    # 9. Linear + all pairwise interactions (n_features <= 6)
    if 2 <= n_features <= 6:
        terms: list[list] = [[mul_prim, _const(1.0), _arg(i)] for i in range(n_features)]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                terms.append([mul_prim, _const(1.0), mul_prim, _arg(i), _arg(j)])
        tokens = terms[0]
        for t in terms[1:]:
            tokens = [add_prim] + tokens + t
        seeds.append([add_prim, const_zero] + tokens)

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
        if np.all(np.isfinite(arr)):
            return arr
        return _safe_evaluate_rows(func, features)
    except Exception:
        return _safe_evaluate_rows(func, features)


# ---- perf #5: check deadline every 64 rows, not every row ------------------
_DEADLINE_CHECK_INTERVAL = 64


def _safe_evaluate_rows(func: object, features: np.ndarray) -> np.ndarray:
    """Row-by-row fallback with NaN for failures.

    Checks the wall-clock deadline every ``_DEADLINE_CHECK_INTERVAL`` rows
    instead of every single row to reduce syscall overhead.
    """
    results = np.empty(features.shape[0], dtype=float)
    deadline = time.monotonic() + _EVAL_ROW_TIMEOUT
    n_rows = features.shape[0]
    for i, row in enumerate(features):
        # perf #5: amortise time.monotonic() over batches of rows
        if i % _DEADLINE_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
            results[i:] = np.nan
            logger.warning(
                "Row-level evaluation timed out after %.1fs; filling %d rows with NaN",
                _EVAL_ROW_TIMEOUT,
                n_rows - i,
            )
            break
        try:
            value = func(*row)  # type: ignore[operator]
            results[i] = np.nan if value is None else float(value)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            results[i] = np.nan
    return results


def make_constant_terminal(value: float) -> gp.Terminal:
    """Create a constant terminal with ``object`` return type."""
    return gp.Terminal(float(value), False, object)


def _compile_cached(individual: gp.PrimitiveTree, pset: gp.PrimitiveSet) -> object:
    # perf #9: LRU eviction — move hit to end, evict from front when full
    key = str(individual)
    func = _COMPILE_CACHE.get(key)
    if func is None:
        func = gp.compile(individual, pset)
        if len(_COMPILE_CACHE) >= _COMPILE_CACHE_MAX:
            # Evict oldest half in one shot to amortise the cost
            evict_count = _COMPILE_CACHE_MAX // 2
            for _ in range(evict_count):
                _COMPILE_CACHE.popitem(last=False)
        _COMPILE_CACHE[key] = func
    else:
        # Move to most-recently-used end
        _COMPILE_CACHE.move_to_end(key)
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
        Wall-clock budget in seconds (default: ``VLP_CONST_OPT_TIMEOUT`` env var,
        default 2.0 s). The Nelder-Mead callback raises ``StopIteration`` when
        the budget is exhausted, preserving the best result found so far.
    """
    _timeout = _CONST_OPT_TIMEOUT if timeout is None else timeout

    indices = [
        idx
        for idx, node in enumerate(individual)
        if isinstance(node, gp.Terminal) and isinstance(node.value, (int, float))
    ]
    if not indices:
        return

    initial = np.array([float(individual[idx].value) for idx in indices], dtype=float)
    inv_n = 1.0 / len(targets)
    diff_buf = np.empty(len(targets), dtype=np.float64)

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
    maxiter = min(50, 10 + 5 * len(indices))
    deadline = time.monotonic() + _timeout

    def _timeout_callback(_intermediate_result) -> None:
        if time.monotonic() > deadline:
            raise StopIteration("optimize_constants timeout")

    try:
        result = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            callback=_timeout_callback,
            options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8, "adaptive": True},
        )
        best = result.x if result.fun < initial_fitness else initial
    except StopIteration:
        logger.debug("optimize_constants interrupted after %.1fs budget", _timeout)
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
    """Evaluate fitness as (penalised_error, tree_complexity)."""
    # perf #4: use module-level constant, not per-call os.environ.get
    try:
        func = _compile_cached(individual, pset)
        preds = vectorised_evaluate(func, features)
        if not np.all(np.isfinite(preds)):
            return _PENALTY_FITNESS
        error = _compute_metric(preds, targets, _FIT_METRIC)
        if not np.isfinite(error):
            return _PENALTY_FITNESS
    except Exception as e:
        logger.exception("Error evaluating individual: %s", e)
        return _PENALTY_FITNESS

    # complexity = float(individual.height) / max_tree_height
    complexity = float(len(individual))
    return error + parsimony_coefficient * complexity, complexity


def migrate(
    islands: list[list[gp.PrimitiveTree]],
    migration_size: int,
    rng: np.random.Generator,
) -> None:
    """Ring-topology migration: each island sends its best to the next island."""
    n = len(islands)
    if n < 2:
        return

    emigrants: list[list[gp.PrimitiveTree]] = []
    for island in islands:
        best = tools.selBest(island, k=min(migration_size, len(island)))
        emigrants.append([deepcopy(ind) for ind in best])

    for i in range(n):
        dest = (i + 1) % n
        dest_island = islands[dest]
        k = min(migration_size, len(dest_island))
        worst = tools.selWorst(dest_island, k=k)
        worst_set = {id(w) for w in worst}
        islands[dest] = [ind for ind in dest_island if id(ind) not in worst_set]
        islands[dest].extend(emigrants[i])

    logger.debug("Migration complete", extra={"islands": n, "migrants_per_island": migration_size})