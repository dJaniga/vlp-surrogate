from __future__ import annotations

import itertools
import logging
import os
import random
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from deap import creator, gp, tools
from scipy.optimize import minimize

from vfp.modeling.tuning_metrics import (
    AVAILABLE_METRICS,
    MAXIMIZE_METRICS,
    evaluate_metric,
)

from .runtime_options import DEFAULT_RUNTIME_OPTIONS, RuntimeOptions

logger = logging.getLogger(__name__)

_PENALTY_FITNESS = (1e18, 1e18)
MINIMIZE_METRICS = AVAILABLE_METRICS.difference(MAXIMIZE_METRICS)

_FIT_METRIC: str = os.environ.get("VLP_FIT_METRIC", "mse")
_CONST_OPT_TIMEOUT: float = float(os.environ.get("VLP_CONST_OPT_TIMEOUT", "2.0"))
_EVAL_ROW_TIMEOUT: float = float(os.environ.get("VLP_EVAL_ROW_TIMEOUT", "5.0"))
_FORCE_SIMPLICITY: bool = (
    os.environ.get("VLP_FORCE_SYMBOLIC_SIMPLICITY", "").lower() == "true"
)

_COMPILE_CACHE: OrderedDict[tuple, object] = OrderedDict()
_COMPILE_CACHE_MAX = 100_000
_DEADLINE_CHECK_INTERVAL = 64


def clear_helper_caches() -> None:
    _COMPILE_CACHE.clear()


def tree_key(individual: gp.PrimitiveTree, runtime: RuntimeOptions) -> tuple:
    if runtime.static_cache_enabled:
        cached = getattr(individual, "_tree_key_cache", None)
        if cached is not None:
            return cached

    out: list = []
    for node in individual:
        if isinstance(node, gp.Terminal):
            out.append(("T", node.name, node.value))
        else:
            out.append(("P", node.name))
    key = tuple(out)

    if runtime.static_cache_enabled:
        try:
            individual._tree_key_cache = key  # type: ignore[attr-defined]
        except AttributeError:
            pass

    return key


def invalidate_individual_caches(individual: gp.PrimitiveTree) -> None:
    for attr in ("_tree_key_cache", "_has_numeric_constants_cache"):
        try:
            delattr(individual, attr)
        except AttributeError:
            pass


def has_numeric_constants(
    individual: gp.PrimitiveTree,
    runtime: RuntimeOptions = DEFAULT_RUNTIME_OPTIONS,
) -> bool:
    if runtime.static_cache_enabled:
        cached = getattr(individual, "_has_numeric_constants_cache", None)
        if cached is not None:
            return bool(cached)

    result = any(
        isinstance(node, gp.Terminal) and isinstance(node.value, (int, float))
        for node in individual
    )

    if runtime.static_cache_enabled:
        try:
            individual._has_numeric_constants_cache = result  # type: ignore[attr-defined]
        except AttributeError:
            pass

    return result


def _compute_metric(preds: np.ndarray, targets: np.ndarray, metric: str) -> float:
    if metric not in MINIMIZE_METRICS:
        raise ValueError(
            f"Invalid metric: {metric} - only {MINIMIZE_METRICS} are supported for minimization"
        )
    return evaluate_metric(metric, preds, targets)


def _c(v: float) -> gp.Terminal:
    return gp.Terminal(v, symbolic=False, ret=float)


def _validate_prefix(tokens: list) -> bool:
    slots = 1
    for tok in tokens:
        if slots <= 0:
            return False
        if isinstance(tok, gp.Primitive):
            slots += tok.arity - 1
        else:
            slots -= 1
    return slots == 0


def _commit(tokens: list, pset: gp.PrimitiveSet, label: str) -> gp.PrimitiveTree | None:
    if not _validate_prefix(tokens):
        logger.warning("Malformed seed skipped: %s", label)
        return None
    try:
        return creator.SymbolicIndividual(tokens)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("Seed build failed (%s): %s", label, exc)
        return None


def _scaled(mul: gp.Primitive, coeff: float, x: gp.Terminal) -> list:
    return [mul, _c(coeff), x]


def _linear(
    add: gp.Primitive,
    mul: gp.Primitive,
    xs: list[gp.Terminal],
    coeffs: list[float] | None = None,
) -> list:
    if not xs:
        raise ValueError("xs must contain at least one terminal")
    if coeffs is None:
        coeffs = [1.0] * len(xs)
    if len(coeffs) != len(xs):
        raise ValueError("coeffs length must match xs length")

    tokens: list = _scaled(mul, coeffs[0], xs[0])
    for c, x in zip(coeffs[1:], xs[1:], strict=False):
        tokens = [add] + tokens + _scaled(mul, c, x)
    return tokens


def _bin(op: gp.Primitive, left: list, right: list) -> list:
    return [op] + left + right


def _un(op: gp.Primitive, inner: list) -> list:
    return [op] + inner


def build_seed_individuals(
    pset: gp.PrimitiveSet,
    n_features: int,
    *,
    include_stochastic: bool = True,
    n_random_seeds: int = 30,
    random_max_depth: int = 6,
    rng: random.Random | None = None,
) -> list[gp.PrimitiveTree]:
    if rng is None:
        rng = random.Random()

    prims: dict[str, gp.Primitive] = {
        p.name: p for prim_group in pset.primitives.values() for p in prim_group
    }
    terms: dict[str, gp.Terminal] = {
        t.name: t for term_group in pset.terminals.values() for t in term_group
    }

    def p(name: str) -> gp.Primitive | None:
        return prims.get(name)

    def a(i: int) -> gp.Terminal:
        return terms[f"ARG{i}"]

    add = p("_add")
    sub = p("_sub")
    mul = p("_mul")
    div = p("_protected_div")
    if add is None or sub is None or mul is None or div is None:
        raise ValueError("Primitive set is missing required arithmetic primitives.")

    sqrt = p("_protected_sqrt")
    square = p("_square")
    abs_p = p("_abs")
    neg = p("_neg")
    args = [a(i) for i in range(n_features)]
    pairs = list(itertools.combinations(range(n_features), 2))
    raw: list[tuple[list, str]] = []

    def s(tokens: list, label: str) -> None:
        raw.append((tokens, label))

    for i, xi in enumerate(args):
        s([xi], f"identity_{i}")
        s(_scaled(mul, 1.0, xi), f"scaled_{i}")
        if square is not None:
            s([square, xi], f"square_{i}")
            s(_bin(add, [square, xi], [xi]), f"square_plus_linear_{i}")
            s(
                _bin(add, [mul, _c(1.0), square, xi], [mul, _c(1.0), xi]),
                f"scaled_square_linear_{i}",
            )
        if abs_p is not None:
            s([abs_p, xi], f"abs_{i}")
        if neg is not None:
            s([neg, xi], f"neg_{i}")
        if sqrt is not None:
            s([sqrt, xi], f"sqrt_{i}")
        if sqrt is not None and square is not None:
            s(_un(sqrt, [square, xi]), f"sqrt_square_{i}")
        if abs_p is not None and square is not None:
            s([square, abs_p, xi], f"square_abs_{i}")
        if neg is not None and square is not None:
            s([neg, square, xi], f"neg_square_{i}")

    for i, j in pairs:
        xi, xj = args[i], args[j]
        s(_bin(add, [xi], [xj]), f"sum_{i}_{j}")
        s(_linear(add, mul, [xi, xj]), f"linear_{i}_{j}")
        s(_bin(sub, [xi], [xj]), f"diff_{i}_{j}")
        s([mul, _c(1.0), sub, xi, xj], f"scaled_diff_{i}_{j}")
        s(_bin(mul, [xi], [xj]), f"product_{i}_{j}")
        s([mul, _c(1.0), mul, xi, xj], f"scaled_product_{i}_{j}")
        s(_bin(div, [xi], [xj]), f"ratio_{i}_{j}")
        s(
            _bin(div, _bin(sub, [xi], [xj]), _bin(add, [xi], [xj])),
            f"signed_norm_diff_{i}_{j}",
        )

        if abs_p is not None:
            s([abs_p, sub, xi, xj], f"abs_diff_{i}_{j}")
            s(_bin(mul, [xi], [abs_p, xj]), f"product_abs_{i}_{j}")
            s([abs_p, mul, xi, xj], f"abs_product_{i}_{j}")
            s(_bin(add, [abs_p, xi], [abs_p, xj]), f"l1_sum_{i}_{j}")
            s(_bin(sub, [abs_p, xi], [abs_p, xj]), f"l1_diff_{i}_{j}")
            s(_bin(mul, [abs_p, xi], [abs_p, xj]), f"l1_product_{i}_{j}")
            s(
                _bin(div, [abs_p, xi], _bin(add, [abs_p, xj], [_c(1e-6)])),
                f"smooth_ratio_{i}_{j}",
            )

        if square is not None:
            s(_bin(sub, [square, xi], [square, xj]), f"diff_squares_{i}_{j}")
            s(_bin(add, [square, xi], [square, xj]), f"sum_squares_{i}_{j}")
            s(_bin(div, [square, xi], [xj]), f"square_ratio_{i}_{j}")

        if sqrt is not None and square is not None:
            s(_un(sqrt, _bin(add, [square, xi], [square, xj])), f"l2_norm_{i}_{j}")
            s(_bin(div, [xi], _un(sqrt, [square, xj])), f"ratio_abs_{i}_{j}")

        if neg is not None:
            s(_bin(add, [xi], [neg, xj]), f"add_neg_{i}_{j}")
            s(_bin(mul, [neg, xi], [xj]), f"neg_product_{i}_{j}")

        if neg is not None and square is not None:
            s(_bin(add, [square, xi], [neg, xj]), f"square_minus_linear_{i}_{j}")

    for i, j in pairs:
        xi, xj = args[i], args[j]
        s(
            _bin(add, _linear(add, mul, [xi, xj]), [mul, _c(1.0), mul, xi, xj]),
            f"linear_interaction_{i}_{j}",
        )

    if square is not None:
        for i, j in pairs:
            xi, xj = args[i], args[j]
            quad_i = _bin(add, [mul, _c(1.0), square, xi], [mul, _c(1.0), xi])
            s(_bin(add, quad_i, [mul, _c(1.0), xj]), f"quadratic_linear_{i}_{j}")

    for i, j, k in list(itertools.combinations(range(n_features), 3))[:6]:
        s([mul, args[i], mul, args[j], args[k]], f"triple_{i}_{j}_{k}")

    if n_features >= 2:
        s(_linear(add, mul, args), "full_linear")

    if n_features >= 3:
        for skip in range(n_features):
            s(
                _linear(add, mul, [args[k] for k in range(n_features) if k != skip]),
                f"leave_out_{skip}",
            )

    if 2 <= n_features <= 6:
        all_terms = [_linear(add, mul, [xi]) for xi in args]
        all_terms.extend([[mul, _c(1.0), mul, args[i], args[j]] for i, j in pairs])
        combined = all_terms[0]
        for term in all_terms[1:]:
            combined = _bin(add, combined, term)
        s(combined, "full_linear_plus_interactions")

    stochastic: list[gp.PrimitiveTree] = []
    if include_stochastic:
        saved = random.getstate()
        random.seed(rng.randint(0, 2**31))
        try:
            for _ in range(n_random_seeds):
                try:
                    expr = gp.genHalfAndHalf(pset, min_=1, max_=random_max_depth)
                    stochastic.append(creator.SymbolicIndividual(expr))  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.debug("Random seed skipped: %s", exc)
        finally:
            random.setstate(saved)

    individuals: list[gp.PrimitiveTree] = []
    seen: set[str] = set()

    for tokens, label in raw:
        ind = _commit(tokens, pset, label)
        if ind is None:
            continue
        key = str(ind)
        if key not in seen:
            seen.add(key)
            individuals.append(ind)

    for ind in stochastic:
        key = str(ind)
        if key not in seen:
            seen.add(key)
            individuals.append(ind)

    logger.info(
        "build_seed_individuals: %d unique seeds (%d stochastic)",
        len(individuals),
        len(stochastic),
    )
    return individuals


def vectorised_evaluate(func: object, features: np.ndarray) -> np.ndarray:
    try:
        n_cols = features.shape[1]
        columns = [features[:, i] for i in range(n_cols)]
        with np.errstate(all="ignore"):
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


def _safe_evaluate_rows(func: object, features: np.ndarray) -> np.ndarray:
    results = np.empty(features.shape[0], dtype=float)
    deadline = time.monotonic() + _EVAL_ROW_TIMEOUT
    n_rows = features.shape[0]

    for i, row in enumerate(features):
        if i % _DEADLINE_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
            results[i:] = np.nan
            logger.warning(
                "Row-level evaluation timed out after %.1fs; filling %d rows with NaN",
                _EVAL_ROW_TIMEOUT,
                n_rows - i,
            )
            break

        try:
            with np.errstate(all="ignore"):
                value = func(*row)  # type: ignore[operator]
            results[i] = np.nan if value is None else float(value)
        except TypeError, ValueError, ZeroDivisionError, OverflowError:
            results[i] = np.nan

    return results


def make_constant_terminal(value: float) -> gp.Terminal:
    return gp.Terminal(float(value), False, object)


def _compile_cached(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    *,
    runtime: RuntimeOptions = DEFAULT_RUNTIME_OPTIONS,
) -> object:
    if not runtime.compile_cache_enabled:
        return gp.compile(individual, pset)

    key = tree_key(individual, runtime)
    func = _COMPILE_CACHE.get(key)

    if func is None:
        func = gp.compile(individual, pset)
        if len(_COMPILE_CACHE) >= _COMPILE_CACHE_MAX:
            evict_count = _COMPILE_CACHE_MAX // 2
            for _ in range(evict_count):
                _COMPILE_CACHE.popitem(last=False)
        _COMPILE_CACHE[key] = func
    else:
        _COMPILE_CACHE.move_to_end(key)

    return func


def optimize_constants(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    timeout: float | None = None,
    sample_size: int | None = None,
    max_constants: int | None = None,
) -> bool:
    opt_timeout = _CONST_OPT_TIMEOUT if timeout is None else timeout

    indices = [
        idx
        for idx, node in enumerate(individual)
        if isinstance(node, gp.Terminal) and isinstance(node.value, (int, float))
    ]

    if not indices:
        return False

    if max_constants is not None and len(indices) > max_constants:
        return False

    opt_features = features
    opt_targets = targets

    if sample_size is not None and 0 < sample_size < len(targets):
        rng = np.random.default_rng(17)
        sample_idx = rng.choice(len(targets), size=sample_size, replace=False)
        opt_features = features[sample_idx]
        opt_targets = targets[sample_idx]

    initial = np.array([float(individual[idx].value) for idx in indices], dtype=float)
    inv_n = 1.0 / len(opt_targets)
    diff_buf = np.empty(len(opt_targets), dtype=np.float64)

    def objective(constants: np.ndarray) -> float:
        for idx, value in zip(indices, constants, strict=False):
            individual[idx] = make_constant_terminal(float(value))

        try:
            func = gp.compile(individual, pset)
            preds = vectorised_evaluate(func, opt_features)
        except Exception:
            return 1e18

        if not np.isfinite(preds).all():
            return 1e18

        np.subtract(preds, opt_targets, out=diff_buf)
        return float(np.dot(diff_buf, diff_buf) * inv_n)

    initial_fitness = objective(initial)

    maxiter = min(50, 10 + 5 * len(indices))
    if len(opt_targets) > 10_000:
        maxiter = min(maxiter, 15)
    elif len(opt_targets) > 2_000:
        maxiter = min(maxiter, 25)

    deadline = time.monotonic() + opt_timeout

    def timeout_callback(_intermediate_result) -> None:
        if time.monotonic() > deadline:
            raise StopIteration("optimize_constants timeout")

    try:
        result = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            callback=timeout_callback,
            options={
                "maxiter": maxiter,
                "xatol": 1e-6,
                "fatol": 1e-8,
                "adaptive": True,
            },
        )
        best = result.x if result.fun < initial_fitness else initial
        improved = bool(result.fun < initial_fitness)
    except StopIteration:
        logger.debug("optimize_constants interrupted after %.1fs budget", opt_timeout)
        return False

    for idx, value in zip(indices, best, strict=False):
        individual[idx] = make_constant_terminal(float(value))

    invalidate_individual_caches(individual)
    return improved


def evaluate_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    parsimony_coefficient: float,
    max_tree_height: int,
    *,
    runtime: RuntimeOptions = DEFAULT_RUNTIME_OPTIONS,
) -> tuple[float, float]:
    try:
        func = _compile_cached(individual, pset, runtime=runtime)
        preds = vectorised_evaluate(func, features)

        if not np.all(np.isfinite(preds)):
            return _PENALTY_FITNESS

        error = _compute_metric(preds, targets, _FIT_METRIC)
        if not np.isfinite(error):
            return _PENALTY_FITNESS
    except Exception as exc:
        logger.debug("Invalid individual evaluation: %s", exc)
        return _PENALTY_FITNESS

    complexity = float(len(individual))
    return error + parsimony_coefficient * complexity, complexity


def migrate(
    islands: list[list[gp.PrimitiveTree]],
    migration_size: int,
    rng: np.random.Generator,
) -> None:
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

    logger.debug(
        "Migration complete",
        extra={"islands": n, "migrants_per_island": migration_size},
    )
