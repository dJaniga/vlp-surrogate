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
_FORCE_SIMPLICITY: bool = (
    os.environ.get("VLP_FORCE_SYMBOLIC_SIMPLICITY", "").lower() == "true"
)

# ---- perf #9: LRU fitness cache (OrderedDict-based, evicts oldest half) -----
_FITNESS_CACHE_MAX = 100_000
_COMPILE_CACHE: OrderedDict[str, object] = OrderedDict()
_COMPILE_CACHE_MAX = 100_000


def _compute_metric(preds: np.ndarray, targets: np.ndarray, metric: str) -> float:
    if metric not in MINIMIZE_METRICS:
        raise ValueError(
            f"Invalid metric: {metric} - only {MINIMIZE_METRICS} are supported for minimization"
        )
    return evaluate_metric(metric, preds, targets)


# def build_seed_individuals(
#     pset: gp.PrimitiveSet,
#     n_features: int,
# ) -> list[gp.PrimitiveTree]:
#     """Create hand-crafted seed individuals that use multiple features."""
#     prim: dict[str, gp.Primitive] = {
#         p.name: p for prims in pset.primitives.values() for p in prims
#     }
#     term: dict[str, gp.Terminal] = {
#         t.name: t for terms in pset.terminals.values() for t in terms
#     }
#
#     add_prim = prim["_add"]
#     mul_prim = prim["_mul"]
#
#     def _arg(i: int) -> gp.Terminal:
#         return term[f"ARG{i}"]
#
#     def _const(v: float) -> gp.Terminal:
#         return make_constant_terminal(v)
#
#     const_zero = _const(0.0)
#     seeds: list[list[gp.Primitive | gp.Terminal]] = []
#
#     # 1. Linear pairwise: c1*ARGi + c2*ARGj
#     for i in range(n_features):
#         for j in range(i + 1, n_features):
#             seeds.append([add_prim, mul_prim, _const(1.0), _arg(i), mul_prim, _const(1.0), _arg(j)])
#
#     # 2. Linear with all features
#     if n_features >= 2:
#         tokens = [mul_prim, _const(1.0), _arg(0)]
#         for i in range(1, n_features):
#             tokens = [add_prim] + tokens + [mul_prim, _const(1.0), _arg(i)]
#         seeds.append([add_prim, const_zero] + tokens)
#
#     # 3. Quadratic in main feature + linear in other
#     if "_square" in prim:
#         square_prim = prim["_square"]
#         for main in range(min(n_features, 2)):
#             other = 1 - main
#             seeds.append([
#                 add_prim, add_prim,
#                 mul_prim, _const(1.0), _arg(main),
#                 mul_prim, _const(1.0), square_prim, _arg(main),
#                 mul_prim, _const(1.0), _arg(other),
#             ])
#
#     # 4. Product interaction: c * ARG0 * ARG1
#     if n_features >= 2:
#         seeds.append([mul_prim, _const(1.0), mul_prim, _arg(0), _arg(1)])
#
#     # 5. Ratio: ARG0 / ARG1
#     if n_features >= 2:
#         seeds.append([prim["_protected_div"], _arg(0), _arg(1)])
#
#     # 6. Affine pairwise with explicit intercept
#     for i in range(n_features):
#         for j in range(i + 1, n_features):
#             seeds.append([
#                 add_prim, add_prim, _const(0.0),
#                 mul_prim, _const(1.0), _arg(i),
#                 mul_prim, _const(1.0), _arg(j),
#             ])
#
#     # 7. Scaled difference: c * (ARGi - ARGj)
#     if "_sub" in prim:
#         sub_prim = prim["_sub"]
#         for i in range(n_features):
#             for j in range(i + 1, n_features):
#                 seeds.append([mul_prim, _const(1.0), sub_prim, _arg(i), _arg(j)])
#
#     # 8. Leave-one-out linear combination (n_features >= 3)
#     if n_features >= 3:
#         for skip in range(n_features):
#             active = [k for k in range(n_features) if k != skip]
#             tokens = [mul_prim, _const(1.0), _arg(active[0])]
#             for k in active[1:]:
#                 tokens = [add_prim] + tokens + [mul_prim, _const(1.0), _arg(k)]
#             seeds.append([add_prim, const_zero] + tokens)
#
#     # 9. Linear + all pairwise interactions (n_features <= 6)
#     if 2 <= n_features <= 6:
#         terms: list[list] = [[mul_prim, _const(1.0), _arg(i)] for i in range(n_features)]
#         for i in range(n_features):
#             for j in range(i + 1, n_features):
#                 terms.append([mul_prim, _const(1.0), mul_prim, _arg(i), _arg(j)])
#         tokens = terms[0]
#         for t in terms[1:]:
#             tokens = [add_prim] + tokens + t
#         seeds.append([add_prim, const_zero] + tokens)
#
#     individuals: list[gp.PrimitiveTree] = []
#     for token_list in seeds:
#         try:
#             ind = creator.SymbolicIndividual(token_list)  # type: ignore[attr-defined]
#             individuals.append(ind)
#         except Exception:
#             continue
#     return individuals

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _c(v: float) -> gp.Terminal:
    """Inline float constant (not an ephemeral — a fixed literal)."""
    return gp.Terminal(v, symbolic=False, ret=float)


def _validate_prefix(tokens: list) -> bool:
    """
    O(n) prefix-tree arity check.

    Counts how many sub-expression slots remain open.  A well-formed
    prefix sequence starts at 1, never goes negative mid-stream, and
    ends at exactly 0.
    """
    slots = 1
    for tok in tokens:
        if slots <= 0:
            return False
        if isinstance(tok, gp.Primitive):
            slots += tok.arity - 1  # a binary op opens 1 extra slot
        else:
            slots -= 1  # a terminal fills one slot
    return slots == 0


def _commit(
    tokens: list,
    pset: gp.PrimitiveSet,
    label: str,
) -> gp.PrimitiveTree | None:
    """Validate *tokens*, wrap in SymbolicIndividual, return None on failure."""
    if not _validate_prefix(tokens):
        logger.warning("Malformed seed skipped: %s", label)
        return None
    try:
        return creator.SymbolicIndividual(tokens)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Seed build failed (%s): %s", label, exc)
        return None


# ---------------------------------------------------------------------------
# Token-list builders  (pure functions, return prefix token lists)
# ---------------------------------------------------------------------------


def _scaled(mul: gp.Primitive, coeff: float, x: gp.Terminal) -> list:
    """c * x"""
    return [mul, _c(coeff), x]


def _linear(
    add: gp.Primitive,
    mul: gp.Primitive,
    xs: list[gp.Terminal],
    coeffs: list[float] | None = None,
) -> list:
    """
    c0*x0 + c1*x1 + … + c_{n-1}*x_{n-1}   (left-associative, prefix)

    Requires len(xs) >= 1.
    """
    if coeffs is None:
        coeffs = [1.0] * len(xs)
    assert len(coeffs) == len(xs), "coeffs length must match xs length"
    tokens: list = _scaled(mul, coeffs[0], xs[0])
    for c, x in zip(coeffs[1:], xs[1:]):
        tokens = [add] + tokens + _scaled(mul, c, x)
    return tokens


def _bin(op: gp.Primitive, left: list, right: list) -> list:
    """op(left, right) — arity-2 combine."""
    return [op] + left + right


def _un(op: gp.Primitive, inner: list) -> list:
    """op(inner) — arity-1 wrap."""
    return [op] + inner


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_seed_individuals(
    pset: gp.PrimitiveSet,
    n_features: int,
    *,
    include_stochastic: bool = True,
    n_random_seeds: int = 30,
    random_max_depth: int = 6,
    rng: random.Random | None = None,
) -> list[gp.PrimitiveTree]:
    """
    Return a validated, deduplicated list of diverse seed individuals.

    Parameters
    ----------
    pset:
        The DEAP PrimitiveSet for the current run.
    n_features:
        Number of input features (ARG0 … ARG{n_features-1}).
    include_stochastic:
        Append randomly grown trees via ``gp.genHalfAndHalf``.
    n_random_seeds:
        How many random trees to attempt (duplicates are dropped).
    random_max_depth:
        Max depth passed to ``genHalfAndHalf``.
    rng:
        Seeded ``random.Random`` for reproducible stochastic seeds.
    """
    if rng is None:
        rng = random.Random()

    # ------------------------------------------------------------------ #
    # Resolve primitives from pset (None = not available in this run)     #
    # ------------------------------------------------------------------ #
    _prims: dict[str, gp.Primitive] = {
        p.name: p for prims in pset.primitives.values() for p in prims
    }
    _terms: dict[str, gp.Terminal] = {
        t.name: t for terms in pset.terminals.values() for t in terms
    }

    def P(name: str) -> gp.Primitive | None:
        return _prims.get(name)

    def A(i: int) -> gp.Terminal:
        return _terms[f"ARG{i}"]

    # Always present
    add = P("_add")
    assert add is not None, "pset missing _add"
    sub = P("_sub")
    assert sub is not None, "pset missing _sub"
    mul = P("_mul")
    assert mul is not None, "pset missing _mul"
    div = P("_protected_div")
    assert div is not None, "pset missing _protected_div"

    # Optional
    sqrt = P("_protected_sqrt")
    square = P("_square")
    abs_p = P("_abs")
    neg = P("_neg")

    args = [A(i) for i in range(n_features)]
    pairs = list(itertools.combinations(range(n_features), 2))

    raw: list[tuple[list, str]] = []  # (token_list, label)

    def S(tokens: list, label: str) -> None:
        raw.append((tokens, label))

    # ==================================================================
    # A. Single-feature seeds
    # ==================================================================

    for i, xi in enumerate(args):
        # A1. Identity
        S([xi], f"A1_identity_{i}")

        # A2. c * ARGi
        S(_scaled(mul, 1.0, xi), f"A2_scaled_{i}")

        # A3. ARGi²
        if square:
            S([square, xi], f"A3_square_{i}")

        # A4. |ARGi|
        if abs_p:
            S([abs_p, xi], f"A4_abs_{i}")

        # A5. -ARGi
        if neg:
            S([neg, xi], f"A5_neg_{i}")

        # A6. √|ARGi|
        if sqrt:
            S([sqrt, xi], f"A6_sqrt_{i}")

        # A7. ARGi²  + ARGi  (linear + quadratic)
        if square:
            S(_bin(add, [square, xi], [xi]), f"A7_sq_plus_lin_{i}")

        # A8. c*ARGi² + c*ARGi
        if square:
            S(
                _bin(
                    add,
                    _scaled(mul, 1.0, [square, xi][0:]),  # noqa – rebuilt below
                    _scaled(mul, 1.0, xi),
                ),
                "",
            )  # placeholder
            raw.pop()
            S(
                _bin(add, [mul, _c(1.0), square, xi], [mul, _c(1.0), xi]),
                f"A8_scaled_sq_lin_{i}",
            )

        # A9. √(ARGi²)  ≡  |ARGi|  — tests sqrt·square composition
        if sqrt and square:
            S(_un(sqrt, [square, xi]), f"A9_sqrt_square_{i}")

        # A10. |ARGi|²  — abs before squaring (numerically stable)
        if abs_p and square:
            S([square, abs_p, xi], f"A10_square_abs_{i}")

        # A11. -ARGi²
        if neg and square:
            S([neg, square, xi], f"A11_neg_square_{i}")

    # ==================================================================
    # B. Pairwise additive / subtractive
    # ==================================================================

    for i, j in pairs:
        xi, xj = args[i], args[j]

        # B1. ARGi + ARGj
        S(_bin(add, [xi], [xj]), f"B1_sum_{i}_{j}")

        # B2. c*ARGi + c*ARGj
        S(_linear(add, mul, [xi, xj]), f"B2_linear_{i}_{j}")

        # B3. ARGi − ARGj
        S(_bin(sub, [xi], [xj]), f"B3_diff_{i}_{j}")

        # B4. c*(ARGi − ARGj)
        S([mul, _c(1.0), sub, xi, xj], f"B4_scaled_diff_{i}_{j}")

        # B5. |ARGi − ARGj|  — L1 distance
        if abs_p:
            S([abs_p, sub, xi, xj], f"B5_abs_diff_{i}_{j}")

        # B6. √(ARGi² + ARGj²)  — L2 norm
        if sqrt and square:
            S(_un(sqrt, _bin(add, [square, xi], [square, xj])), f"B6_l2_norm_{i}_{j}")

        # B7. ARGi² − ARGj²  — difference of squares
        if square:
            S(_bin(sub, [square, xi], [square, xj]), f"B7_diff_squares_{i}_{j}")

        # B8. ARGi² + ARGj²
        if square:
            S(_bin(add, [square, xi], [square, xj]), f"B8_sum_squares_{i}_{j}")

    # ==================================================================
    # C. Multiplicative / ratio
    # ==================================================================

    for i, j in pairs:
        xi, xj = args[i], args[j]

        # C1. ARGi * ARGj
        S(_bin(mul, [xi], [xj]), f"C1_product_{i}_{j}")

        # C2. c * ARGi * ARGj
        S([mul, _c(1.0), mul, xi, xj], f"C2_scaled_product_{i}_{j}")

        # C3. ARGi / ARGj
        S(_bin(div, [xi], [xj]), f"C3_ratio_{i}_{j}")

        # C4. (ARGi − ARGj) / (ARGi + ARGj)  — signed normalised difference
        S(
            _bin(div, _bin(sub, [xi], [xj]), _bin(add, [xi], [xj])),
            f"C4_signed_norm_diff_{i}_{j}",
        )

        # C5. ARGi² / ARGj
        if square:
            S(_bin(div, [square, xi], [xj]), f"C5_sq_ratio_{i}_{j}")

        # C6. ARGi / √ARGj²  ≡  ARGi / |ARGj|
        if sqrt and square:
            S(_bin(div, [xi], _un(sqrt, [square, xj])), f"C6_ratio_abs_{i}_{j}")

        # C7. ARGi * |ARGj|
        if abs_p:
            S(_bin(mul, [xi], [abs_p, xj]), f"C7_product_abs_{i}_{j}")

        # C8. |ARGi * ARGj|
        if abs_p:
            S([abs_p, mul, xi, xj], f"C8_abs_product_{i}_{j}")

    # ==================================================================
    # D. Linear + interaction  (c*xi + c*xj + c*xi*xj)
    # ==================================================================

    for i, j in pairs:
        xi, xj = args[i], args[j]
        lin = _linear(add, mul, [xi, xj])
        prod = [mul, _c(1.0), mul, xi, xj]
        S(_bin(add, lin, prod), f"D1_lin_inter_{i}_{j}")

    # ==================================================================
    # E. Quadratic expansion in one feature + linear in another
    #    c*xi² + c*xi + c*xj
    # ==================================================================

    if square:
        for i, j in pairs:
            xi, xj = args[i], args[j]
            quad_i = _bin(add, [mul, _c(1.0), square, xi], [mul, _c(1.0), xi])
            lin_j = [mul, _c(1.0), xj]
            S(_bin(add, quad_i, lin_j), f"E1_quadratic_{i}_linear_{j}")

    # ==================================================================
    # F. Triple products  ARGi * ARGj * ARGk
    # ==================================================================

    triples = list(itertools.combinations(range(n_features), 3))
    for i, j, k in triples[:6]:  # cap to avoid explosion
        S([mul, args[i], mul, args[j], args[k]], f"F1_triple_{i}_{j}_{k}")

    # ==================================================================
    # G. Full linear combination of all features
    # ==================================================================

    if n_features >= 2:
        S(_linear(add, mul, args), "G1_full_linear")

    # ==================================================================
    # H. Leave-one-out linear  (n_features >= 3)
    # ==================================================================

    if n_features >= 3:
        for skip in range(n_features):
            active = [args[k] for k in range(n_features) if k != skip]
            S(_linear(add, mul, active), f"H1_leave_out_{skip}")

    # ==================================================================
    # I. Full linear + all pairwise interactions  (2 ≤ n ≤ 6)
    # ==================================================================

    if 2 <= n_features <= 6:
        lin_terms = [_linear(add, mul, [xi]) for xi in args]
        prod_terms = [[mul, _c(1.0), mul, args[i], args[j]] for i, j in pairs]
        all_terms = lin_terms + prod_terms
        combined = all_terms[0]
        for t in all_terms[1:]:
            combined = _bin(add, combined, t)
        S(combined, "I1_full_linear_plus_interactions")

    # ==================================================================
    # J. Sqrt-based seeds (exploiting _protected_sqrt more deeply)
    # ==================================================================

    if sqrt:
        for i, j in pairs[:4]:
            xi, xj = args[i], args[j]

            # J1. √|xi| + √|xj|
            if abs_p:
                S(
                    _bin(add, _un(sqrt, [abs_p, xi]), _un(sqrt, [abs_p, xj])),
                    f"J1_sqrt_abs_sum_{i}_{j}",
                )

            # J2. √(xi² + xj²) + c*(xi - xj)  — norm + offset
            if square and sub:
                norm = _un(sqrt, _bin(add, [square, xi], [square, xj]))
                diff = [mul, _c(1.0), sub, xi, xj]
                S(_bin(add, norm, diff), f"J2_norm_plus_diff_{i}_{j}")

            # J3. xi / √(xj² + c)  — soft normalisation
            if square:
                denom = _un(sqrt, _bin(add, [square, xj], [_c(1e-6)]))
                # _c(1e-6) is a Terminal, not a list — build correctly:
                denom = [sqrt, add, square, xj, _c(1e-6)]
                S(_bin(div, [xi], denom), f"J3_soft_norm_{i}_{j}")

    # ==================================================================
    # K. Abs-based seeds
    # ==================================================================

    if abs_p:
        for i, j in pairs[:4]:
            xi, xj = args[i], args[j]

            # K1. |xi| + |xj|  — L1 sum
            S(_bin(add, [abs_p, xi], [abs_p, xj]), f"K1_l1_sum_{i}_{j}")

            # K2. |xi| − |xj|
            S(_bin(sub, [abs_p, xi], [abs_p, xj]), f"K2_l1_diff_{i}_{j}")

            # K3. |xi| * |xj|
            S(_bin(mul, [abs_p, xi], [abs_p, xj]), f"K3_l1_product_{i}_{j}")

            # K4. |xi| / (|xj| + c)  — smooth ratio
            denom_k4 = _bin(add, [abs_p, xj], [_c(1e-6)])
            S(_bin(div, [abs_p, xi], denom_k4), f"K4_smooth_ratio_{i}_{j}")

    # ==================================================================
    # L. Neg-based seeds
    # ==================================================================

    if neg:
        for i, j in pairs[:4]:
            xi, xj = args[i], args[j]

            # L1. xi + (−xj)  ≡  xi − xj  (different tree topology)
            S(_bin(add, [xi], [neg, xj]), f"L1_add_neg_{i}_{j}")

            # L2. (−xi) * xj
            S(_bin(mul, [neg, xi], [xj]), f"L2_neg_product_{i}_{j}")

            # L3. xi² − xj  (quadratic minus linear)
            if square:
                S(_bin(add, [square, xi], [neg, xj]), f"L3_sq_minus_lin_{i}_{j}")

    # ==================================================================
    # Stochastic diversity via genHalfAndHalf
    # ==================================================================

    stochastic: list[gp.PrimitiveTree] = []
    if include_stochastic:
        saved = random.getstate()
        random.seed(rng.randint(0, 2**31))
        for _ in range(n_random_seeds):
            try:
                expr = gp.genHalfAndHalf(pset, min_=1, max_=random_max_depth)
                ind = creator.SymbolicIndividual(expr)  # type: ignore[attr-defined]
                stochastic.append(ind)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Random seed skipped: %s", exc)
        random.setstate(saved)

    # ==================================================================
    # Validate, deduplicate, assemble
    # ==================================================================

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
        "build_seed_individuals: %d unique seeds "
        "(%d hand-crafted validated, %d stochastic)",
        len(individuals),
        len(individuals) - len(stochastic),
        len(stochastic),
    )

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
        except TypeError, ValueError, ZeroDivisionError, OverflowError:
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
            logger.info("LRU compile cache full, evicting oldest half")
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
            options={
                "maxiter": maxiter,
                "xatol": 1e-6,
                "fatol": 1e-8,
                "adaptive": True,
            },
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

    logger.debug(
        "Migration complete",
        extra={"islands": n, "migrants_per_island": migration_size},
    )
