from __future__ import annotations

from copy import deepcopy
import logging

import numpy as np
from deap import creator, gp, tools
from scipy.optimize import minimize


logger = logging.getLogger(__name__)

# TODO: paramterize
_PENALTY_FITNESS = (1e18, 1e18)


def build_seed_individuals(
    pset: gp.PrimitiveSet,
    n_features: int,
) -> list[gp.PrimitiveTree]:
    """Create hand-crafted seed individuals that use multiple features.

    These provide the GP with multi-variable starting structures that it
    can refine via crossover, mutation, and constant optimisation.
    """
    # Look up primitives and terminals from the pset
    prim: dict[str, gp.Primitive] = {}
    for prims in pset.primitives.values():
        for p in prims:
            prim[p.name] = p

    term: dict[str, gp.Terminal] = {}
    for terms in pset.terminals.values():
        for t in terms:
            term[t.name] = t

    def _arg(i: int) -> gp.Terminal:
        return term[f"ARG{i}"]

    def _const(v: float) -> gp.Terminal:
        return make_constant_terminal(v)

    seeds: list[list[gp.Primitive | gp.Terminal]] = []

    # 1. Linear: c0 + c1*ARG0 + c2*ARG1  (all pairwise combos of 2 features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # _add(_mul(c1, ARGi), _mul(c2, ARGj))
            tokens = [
                prim["_add"],
                prim["_mul"],
                _const(1.0),
                _arg(i),
                prim["_mul"],
                _const(1.0),
                _arg(j),
            ]
            seeds.append(tokens)

    # 2. Linear with all features: c0 + c1*ARG0 + c2*ARG1 + c3*ARG2 + c4*ARG3
    if n_features >= 2:
        # Build: _add(_add(...), _mul(c, ARGn))
        # Start with _mul(c, ARG0)
        tokens = [prim["_mul"], _const(1.0), _arg(0)]
        for i in range(1, n_features):
            tokens = [prim["_add"]] + tokens + [prim["_mul"], _const(1.0), _arg(i)]
        # Add a constant offset: _add(c0, <above>)
        tokens = [prim["_add"], _const(0.0)] + tokens
        seeds.append(tokens)

    # 3. Quadratic in main feature + linear in others:
    #    c0 + c1*ARG0 + c2*ARG0^2 + c3*ARG1
    for main in range(min(n_features, 2)):
        other = 1 - main
        tokens = [
            prim["_add"],
            prim["_add"],
            prim["_mul"],
            _const(1.0),
            _arg(main),
            prim["_mul"],
            _const(1.0),
            prim["_square"],
            _arg(main),
            prim["_mul"],
            _const(1.0),
            _arg(other),
        ]
        seeds.append(tokens)

    # 4. Product interaction: c * ARG0 * ARG1
    if n_features >= 2:
        tokens = [
            prim["_mul"],
            _const(1.0),
            prim["_mul"],
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

    individuals: list[gp.PrimitiveTree] = []
    for token_list in seeds:
        try:
            ind = creator.SymbolicIndividual(token_list)  # type: ignore[attr-defined]
            individuals.append(ind)
        except Exception:
            continue

    logger.debug(
        "Seed individuals created",
        extra={"count": len(individuals), "n_features": n_features},
    )
    return individuals


def vectorised_evaluate(func: object, features: np.ndarray) -> np.ndarray:
    """Evaluate a compiled GP function over all feature rows."""
    try:
        columns = [features[:, i] for i in range(features.shape[1])]
        result = func(*columns)  # type: ignore[operator]
        if result is None:
            return _safe_evaluate_rows(func, features)
        arr = np.asarray(result, dtype=float)
        if arr.shape != (features.shape[0],):
            # scalar result (constant tree) -- broadcast
            arr = np.full(features.shape[0], float(arr), dtype=float)
        if np.all(np.isfinite(arr)):
            return arr
        # some non-finite values, fall back
        return _safe_evaluate_rows(func, features)
    except Exception:
        return _safe_evaluate_rows(func, features)


def ectorised_evaluate(func: object, features: np.ndarray) -> np.ndarray:
    """Evaluate a compiled GP function over all feature rows."""
    try:
        columns = [features[:, i] for i in range(features.shape[1])]
        result = func(*columns)  # type: ignore[operator]
        if result is None:
            return _safe_evaluate_rows(func, features)
        arr = np.asarray(result, dtype=float)
        if arr.shape != (features.shape[0],):
            # scalar result (constant tree) -- broadcast
            arr = np.full(features.shape[0], float(arr), dtype=float)
        if np.all(np.isfinite(arr)):
            return arr
        # some non-finite values, fall back
        return _safe_evaluate_rows(func, features)
    except Exception:
        return _safe_evaluate_rows(func, features)


def _safe_evaluate_rows(func: object, features: np.ndarray) -> np.ndarray:
    """Row-by-row fallback with NaN for failures."""
    results = np.empty(features.shape[0], dtype=float)
    for i, row in enumerate(features):
        try:
            value = func(*row)  # type: ignore[operator]
            if value is None:
                results[i] = np.nan
            else:
                results[i] = float(value)
        except TypeError, ValueError, ZeroDivisionError, OverflowError:
            results[i] = np.nan
    return results


def make_constant_terminal(value: float) -> gp.Terminal:
    """Create a constant terminal with ``object`` return type (matches the untyped pset)."""
    # The value is explicitly cast to a Python float so that repr(value)
    # produces a plain literal like 1.0 rather than np.float64(1.0).
    return gp.Terminal(float(value), False, object)


def optimize_constants(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Fine-tune ephemeral constants in the individual via Nelder-Mead."""
    indices = [
        idx
        for idx, node in enumerate(individual)
        if isinstance(node, gp.Terminal) and isinstance(node.value, (int, float))
    ]
    if not indices:
        return

    initial = np.array([float(individual[idx].value) for idx in indices], dtype=float)

    def objective(constants: np.ndarray) -> float:
        for idx, value in zip(indices, constants, strict=True):
            individual[idx] = make_constant_terminal(value)
        func = gp.compile(individual, pset)
        preds = vectorised_evaluate(func, features)
        if np.any(~np.isfinite(preds)):
            return 1e18
        return float(np.mean((preds - targets) ** 2))

    result = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 100, "xatol": 1e-6, "fatol": 1e-8},
    )
    # Comparing against initial MSE to avoid regressing
    best = result.x
    best_mse = objective(best)
    initial_mse = objective(initial)
    if initial_mse < best_mse:
        best = initial
    for idx, value in zip(indices, best, strict=True):
        individual[idx] = make_constant_terminal(value)

    logger.debug(
        "Constants optimized",
        extra={"count": len(indices), "success": result.success},
    )


def evaluate_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    parsimony_coefficient: float = 0.0,
) -> tuple[float, float]:
    """Evaluate fitness as (MSE + parsimony_penalty, tree_length).

    The parsimony term penalises bloated trees even when their raw MSE is
    competitive.
    """
    try:
        func = gp.compile(individual, pset)
        preds = vectorised_evaluate(func, features)
        if np.any(~np.isfinite(preds)):
            return _PENALTY_FITNESS
        mse = float(np.mean((preds - targets) ** 2))
    except Exception:
        return _PENALTY_FITNESS

    complexity = float(len(individual))
    penalised_mse = mse + parsimony_coefficient * complexity
    return penalised_mse, complexity


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
    emigrants: list[list[gp.PrimitiveTree]] = []
    for island in islands:
        best = tools.selBest(island, k=min(migration_size, len(island)))
        emigrants.append([deepcopy(ind) for ind in best])

    for i in range(n):
        dest = (i + 1) % n
        dest_island = islands[dest]
        worst = tools.selWorst(dest_island, k=min(migration_size, len(dest_island)))
        for w in worst:
            if w in dest_island:
                dest_island.remove(w)
        dest_island.extend(emigrants[i])

    logger.debug(
        "Migration complete",
        extra={"islands": n, "migrants_per_island": migration_size},
    )
