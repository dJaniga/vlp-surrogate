from __future__ import annotations

import concurrent.futures
import functools
import logging
from typing import Callable

import numpy as np
import sympy
from deap import creator, gp

from .helpers import make_constant_terminal, evaluate_individual

logger = logging.getLogger(__name__)

# TODO: parameterize
_PENALTY_FITNESS = (1e18, 1e18)

# Timeout (seconds) for the heavy SymPy pipeline.
_SIMPLIFY_TIMEOUT: float = 5.0

# Expressions at or below this op-count run inline (no thread/executor overhead);
# the SymPy work for them is microseconds, so the submit()/result() round-trip of
# the executor would dominate.
_INLINE_OP_THRESHOLD: int = 8

_SYMPY_SYMBOLS: dict[int, list[sympy.Symbol]] = {}
_SYM_NAME_MAP: dict[int, dict[str, str]] = {}

# A couple of GIL-bound SymPy worker threads is plenty; more than that just
# thrashes under the GIL while one heavy call holds it.
_SIMPLIFY_EXECUTOR: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(max_workers=2)
)

# Resolve a couple of hot attribute lookups once at import time.
_Float = sympy.Float
_count_ops = sympy.count_ops


def _get_sympy_symbols(n: int) -> list[sympy.Symbol]:
    """Return a cached list of SymPy symbols ARG0 .. ARG(n-1)."""
    syms = _SYMPY_SYMBOLS.get(n)
    if syms is None:
        syms = [sympy.Symbol(f"ARG{i}") for i in range(n)]
        _SYMPY_SYMBOLS[n] = syms
        _SYM_NAME_MAP[n] = {s.name: f"ARG{i}" for i, s in enumerate(syms)}
    return syms


# Mapping from primitive names to SymPy equivalents.
_SYMPY_OP_MAP: dict[str, Callable[..., sympy.Expr]] | None = None


def _sympy_op_map() -> dict[str, Callable[..., sympy.Expr]]:
    global _SYMPY_OP_MAP
    if _SYMPY_OP_MAP is None:
        _SYMPY_OP_MAP = {
            "_add": lambda a, b: a + b,
            "_sub": lambda a, b: a - b,
            "_mul": lambda a, b: a * b,
            "_protected_div": lambda a, b: a / b,
            "_square": lambda a: a**2,
            "_neg": lambda a: -a,
            "_abs": lambda a: sympy.Abs(a),
            "_protected_sqrt": lambda a: sympy.sqrt(sympy.Abs(a)),
        }
    return _SYMPY_OP_MAP


# Cache pset lookup dicts keyed by id(pset) — pset never changes during a pass.
_PSET_LOOKUP_CACHE: dict[
    int, tuple[dict[str, gp.Primitive], dict[str, gp.Terminal]]
] = {}


def _get_pset_lookups(
    pset: gp.PrimitiveSet,
) -> tuple[dict[str, gp.Primitive], dict[str, gp.Terminal]]:
    """Return cached (prim_by_name, term_by_name) dicts for the given pset."""
    key = id(pset)
    cached = _PSET_LOOKUP_CACHE.get(key)
    if cached is None:
        prim_by_name: dict[str, gp.Primitive] = {
            p.name: p for prims in pset.primitives.values() for p in prims
        }
        term_by_name: dict[str, gp.Terminal] = {
            t.name: t for terms in pset.terminals.values() for t in terms
        }
        cached = (prim_by_name, term_by_name)
        _PSET_LOOKUP_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Structural key: cheap, hashable fingerprint of a DEAP tree. Two trees with
# the same key are byte-for-byte equivalent, so we can memoize the whole
# simplification result on it.
# ---------------------------------------------------------------------------
def _structural_key(individual: gp.PrimitiveTree) -> tuple:
    out: list = []
    for node in individual:
        if isinstance(node, gp.Terminal):
            out.append((0, node.name, node.value))
        else:
            out.append((1, node.name))
    return tuple(out)


def _deap_to_sympy(
    individual: gp.PrimitiveTree,
    n_features: int,
) -> sympy.Expr | None:
    """Convert a DEAP PrimitiveTree into a SymPy expression.

    Returns ``None`` if the conversion fails (unsupported nodes, etc.).
    """
    symbols = _get_sympy_symbols(n_features)
    op_map = _sympy_op_map()

    stack: list[sympy.Expr] = []
    # Walk the tree in reverse (postfix evaluation).
    for node in reversed(individual):
        if isinstance(node, gp.Terminal):
            name = node.name
            if name.startswith("ARG"):
                stack.append(symbols[int(name[3:])])
            else:
                # ephemeral constant
                try:
                    stack.append(sympy.Float(float(node.value)))
                except (TypeError, ValueError):
                    return None
        elif isinstance(node, gp.Primitive):
            fn = op_map.get(node.name)
            if fn is None:
                return None
            args = [stack.pop() for _ in range(node.arity)]
            try:
                stack.append(fn(*args))
            except Exception:
                return None
        else:
            return None

    if len(stack) != 1:
        return None
    return stack[0]


_SYMPY_TO_DEAP_BINARY = {
    sympy.Add: "_add",
    sympy.Mul: "_mul",
}


def _sympy_to_deap_tokens(
    expr: sympy.Expr,
    pset: gp.PrimitiveSet,
    n_features: int,
) -> list[gp.Primitive | gp.Terminal] | None:
    """Convert a SymPy expression into a flat list of DEAP GP tokens (prefix order).

    Returns ``None`` if the expression cannot be faithfully represented in the
    primitive set (e.g. unsupported SymPy operations).
    """
    _get_sympy_symbols(n_features)  # ensures name map is populated
    sym_name_map = _SYM_NAME_MAP[n_features]
    prim_by_name, term_by_name = _get_pset_lookups(pset)

    tokens: list[gp.Primitive | gp.Terminal] = []

    def _walk(e: sympy.Expr) -> bool:
        # Symbol (ARGi)
        if isinstance(e, sympy.Symbol):
            name = sym_name_map.get(e.name)
            term = term_by_name.get(name) if name is not None else None
            if term is None:
                return False
            tokens.append(term)
            return True

        # Numeric constant
        if isinstance(
            e,
            (
                sympy.Number,
                sympy.core.numbers.Float,
                sympy.core.numbers.Integer,
                sympy.core.numbers.Rational,
            ),
        ):
            tokens.append(make_constant_terminal(float(e)))
            return True

        if isinstance(e, sympy.Abs):
            prim = prim_by_name.get("_abs")
            if prim is None:
                return False
            tokens.append(prim)
            return _walk(e.args[0])

        # Pow (x**2 -> _square, x**0.5 -> _protected_sqrt, x**-1 -> _protected_div(1,x))
        if isinstance(e, sympy.Pow):
            base_expr, exp_expr = e.args
            if exp_expr == 2:
                prim = prim_by_name.get("_square")
                if prim is None:
                    return False
                tokens.append(prim)
                return _walk(base_expr)
            if exp_expr == sympy.Rational(1, 2) or exp_expr == sympy.Float(0.5):
                prim = prim_by_name.get("_protected_sqrt")
                if prim is None:
                    return False
                tokens.append(prim)
                return _walk(base_expr)
            if exp_expr == -1:
                # x**-1 -> _protected_div(1, x)
                prim = prim_by_name.get("_protected_div")
                if prim is None:
                    return False
                tokens.append(prim)
                tokens.append(make_constant_terminal(1.0))
                return _walk(base_expr)
            # general power -- cannot represent
            return False

        # Add / Mul (n-ary -> chained binary)
        prim_name = _SYMPY_TO_DEAP_BINARY.get(type(e))
        if prim_name is not None:
            prim = prim_by_name.get(prim_name)
            if prim is None:
                return False
            args = e.args
            if len(args) < 2:
                return _walk(args[0]) if len(args) == 1 else False
            # chain: op(a, op(b, op(c, d)))
            for _ in range(len(args) - 1):
                tokens.append(prim)
            for sub in args[:-1]:
                if not _walk(sub):
                    return False
            return _walk(args[-1])

        return False

    if not _walk(expr):
        return None
    return tokens


def _has_radical(expr: sympy.Expr) -> bool:
    """True iff expr contains a non-integer rational power (a radical that
    radsimp could rationalize)."""
    return any(
        p.exp.is_Rational and not p.exp.is_Integer for p in expr.atoms(sympy.Pow)
    )


def _do_sympy_simplify(expr: sympy.Expr) -> sympy.Expr:
    """Run the algebraic SymPy simplification pipeline."""
    if expr.has(_Float):
        result = sympy.nsimplify(expr, rational=False, tolerance=1e-8)
    else:
        result = expr

    if _has_radical(result):
        result = sympy.radsimp(result)
    result = sympy.cancel(result)

    op_count = _count_ops(result)

    if op_count <= 3:
        return result

    if op_count <= 30:
        candidate = sympy.factor(sympy.expand(result))
        return candidate if _count_ops(candidate) < op_count else result

    # Heavy path: a direct factor() of the already-cancelled form is the genuine
    # algebraic-only step and far cheaper than full simplify().
    candidate = sympy.factor(result)
    return candidate if _count_ops(candidate) < op_count else result


def _simplify_expr(expr: sympy.Expr) -> sympy.Expr | None:
    """Simplify `expr`, running inline for cheap expressions and on a worker
    thread (with a wall-clock timeout) for the expensive ones.

    NOTE: the timeout frees *us* to move on; the GIL-bound SymPy call may keep
    running in the background, which is why we cap the pool at 2 workers.
    """
    if _count_ops(expr) <= _INLINE_OP_THRESHOLD:
        try:
            return _do_sympy_simplify(expr)
        except Exception:
            return None

    future = _SIMPLIFY_EXECUTOR.submit(_do_sympy_simplify, expr)
    try:
        return future.result(timeout=_SIMPLIFY_TIMEOUT)
    except concurrent.futures.TimeoutError:
        logger.debug("SymPy simplification timed out, skipping individual")
        future.cancel()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result memoization keyed on the structural fingerprint of the DEAP tree.
# This is the cache that actually pays off: identical individuals recur across
# generations and islands within a single fit.
#   value = list[token] (simplified, shorter)  OR  None (not simplifiable)
# ---------------------------------------------------------------------------
_RESULT_CACHE: dict[tuple, list[gp.Primitive | gp.Terminal] | None] = {}
_RESULT_CACHE_MAX = 50_000


def _compute_simplified_tokens(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    n_features: int,
) -> list[gp.Primitive | gp.Terminal] | None:
    """Pure structural simplification (no fitness check). Returns shorter token
    list, or None if it can't be simplified / isn't shorter."""
    sym_expr = _deap_to_sympy(individual, n_features)
    if sym_expr is None:
        return None

    simplified = _simplify_expr(sym_expr)
    if simplified is None:
        return None

    new_tokens = _sympy_to_deap_tokens(simplified, pset, n_features)
    if new_tokens is None or len(new_tokens) >= len(individual):
        return None
    return new_tokens


def _simplify_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    n_features: int,
    parsimony_coefficient: float,
    max_tree_height: int,
) -> bool:
    """Attempt to algebraically simplify an individual in-place.

    Returns True if simplification was applied, False otherwise.
    """
    original_len = len(individual)
    if original_len <= 3:
        return False

    key = _structural_key(individual)
    if key in _RESULT_CACHE:
        new_tokens = _RESULT_CACHE[key]
        if new_tokens is None:
            return False
    else:
        new_tokens = _compute_simplified_tokens(individual, pset, n_features)
        if len(_RESULT_CACHE) < _RESULT_CACHE_MAX:
            _RESULT_CACHE[key] = new_tokens
        if new_tokens is None:
            return False

    # Validate fitness before accepting (depends on data, so not cacheable).
    try:
        new_individual = creator.SymbolicIndividual(new_tokens)  # type: ignore[attr-defined]
        new_fitness = evaluate_individual(
            new_individual,
            pset,
            features,
            targets,
            parsimony_coefficient,
            max_tree_height,
        )
        if new_fitness[0] >= _PENALTY_FITNESS[0]:
            return False
        # Only accept if it does not significantly degrade the error.
        if new_fitness[0] > individual.fitness.values[0] + 1e-6:  # type: ignore[attr-defined]
            return False
    except Exception:
        return False

    individual[0:len(individual)] = new_tokens
    individual.fitness.values = new_fitness  # type: ignore[attr-defined]

    logger.debug(
        "Individual simplified",
        extra={"before": original_len, "after": len(new_tokens)},
    )
    return True


def simplify_island(
    island: list[gp.PrimitiveTree],
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    n_features: int,
    parsimony_coefficient: float,
    max_tree_height: int,
) -> None:
    """Apply SymPy simplification to all individuals in an island."""
    simplified_count = 0
    for individual in island:
        if _simplify_individual(
            individual,
            pset,
            features,
            targets,
            n_features,
            parsimony_coefficient,
            max_tree_height,
        ):
            simplified_count += 1
    if simplified_count > 0:
        logger.debug(
            "Island simplification pass",
            extra={"simplified": simplified_count, "total": len(island)},
        )