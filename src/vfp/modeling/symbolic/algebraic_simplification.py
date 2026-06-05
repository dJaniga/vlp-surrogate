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

# TODO: paramterize
_PENALTY_FITNESS = (1e18, 1e18)

# Timeout (seconds) for the heavy sympy.simplify call
_SIMPLIFY_TIMEOUT: float = 5.0

_SYMPY_SYMBOLS: dict[int, list[sympy.Symbol]] = {}

_SIMPLIFY_EXECUTOR: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(max_workers=10)
)


def _get_sympy_symbols(n: int) -> list[sympy.Symbol]:
    """Return a cached list of SymPy symbols ARG0 .. ARG(n-1)."""
    if n not in _SYMPY_SYMBOLS:
        _SYMPY_SYMBOLS[n] = [sympy.Symbol(f"ARG{i}") for i in range(n)]
    return _SYMPY_SYMBOLS[n]


# Mapping from primitive names to SymPy equivalents.
# Built lazily so the module can be imported without SymPy if unused.
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


# Fix 3: Cache pset lookup dicts keyed by id(pset) — pset never changes during a pass.
_PSET_LOOKUP_CACHE: dict[
    int, tuple[dict[str, gp.Primitive], dict[str, gp.Terminal]]
] = {}


def _get_pset_lookups(
    pset: gp.PrimitiveSet,
) -> tuple[dict[str, gp.Primitive], dict[str, gp.Terminal]]:
    """Return cached (prim_by_name, term_by_name) dicts for the given pset."""
    key = id(pset)
    if key not in _PSET_LOOKUP_CACHE:
        prim_by_name: dict[str, gp.Primitive] = {
            p.name: p for prims in pset.primitives.values() for p in prims
        }
        term_by_name: dict[str, gp.Terminal] = {
            t.name: t for terms in pset.terminals.values() for t in terms
        }
        _PSET_LOOKUP_CACHE[key] = (prim_by_name, term_by_name)
    return _PSET_LOOKUP_CACHE[key]


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
    # Walk the tree in reverse (postfix evaluation)
    for node in reversed(individual):
        if isinstance(node, gp.Terminal):
            if node.name.startswith("ARG"):
                idx = int(node.name[3:])
                stack.append(symbols[idx])
            else:
                # ephemeral constant
                try:
                    stack.append(sympy.Float(float(node.value)))
                except TypeError, ValueError:
                    return None
        elif isinstance(node, gp.Primitive):
            if node.name not in op_map:
                return None
            args = [stack.pop() for _ in range(node.arity)]
            try:
                result = op_map[node.name](*args)
                stack.append(result)
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
    symbols = _get_sympy_symbols(n_features)
    sym_name_map = {s.name: f"ARG{i}" for i, s in enumerate(symbols)}

    # Fix 3: use cached lookups instead of rebuilding dicts on every call
    prim_by_name, term_by_name = _get_pset_lookups(pset)

    tokens: list[gp.Primitive | gp.Terminal] = []

    def _walk(e: sympy.Expr) -> bool:
        # Symbol  (ARGi)
        if isinstance(e, sympy.Symbol):
            name = sym_name_map.get(e.name)
            if name is None or name not in term_by_name:
                return False
            tokens.append(term_by_name[name])
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
            if "_abs" not in prim_by_name:
                return False
            tokens.append(prim_by_name["_abs"])
            return _walk(e.args[0])

        # Pow  (x**2 -> _square,  x**0.5 -> _protected_sqrt,  x**-1 -> _protected_div(1,x))
        if isinstance(e, sympy.Pow):
            base_expr, exp_expr = e.args
            if exp_expr == 2:
                if "_square" not in prim_by_name:
                    return False
                tokens.append(prim_by_name["_square"])
                return _walk(base_expr)
            if exp_expr == sympy.Rational(1, 2) or exp_expr == sympy.Float(0.5):
                if "_protected_sqrt" not in prim_by_name:
                    return False
                tokens.append(prim_by_name["_protected_sqrt"])
                return _walk(base_expr)
            if exp_expr == -1:
                # x**-1  ->  _protected_div(1, x)
                if "_protected_div" not in prim_by_name:
                    return False
                tokens.append(prim_by_name["_protected_div"])
                tokens.append(make_constant_terminal(1.0))
                return _walk(base_expr)
            # general power -- cannot represent
            return False

        # Add / Mul  (n-ary -> chained binary)
        if type(e) in _SYMPY_TO_DEAP_BINARY:
            prim_name = _SYMPY_TO_DEAP_BINARY[type(e)]
            if prim_name not in prim_by_name:
                return False
            args = list(e.args)
            if len(args) < 2:
                if len(args) == 1:
                    return _walk(args[0])
                return False
            # chain: op(a, op(b, op(c, d)))
            # first emit (n-1) ops, then the leaves
            for _ in range(len(args) - 1):
                tokens.append(prim_by_name[prim_name])
            # walk first arg
            if not _walk(args[0]):
                return False
            # recurse on remaining, chaining
            for i in range(1, len(args) - 1):
                if not _walk(args[i]):
                    return False
            return _walk(args[-1])

        # Negation:  -x  appears as Mul(-1, x)
        # Subtraction doesn't exist in SymPy -- it's Add(a, Mul(-1, b))
        # We try to detect it: if we see Add(a, Mul(-1, b)), emit _sub(a,b)
        return False

    if not _walk(expr):
        return None
    return tokens


@functools.lru_cache(maxsize=512)
def _do_sympy_simplify(expr: sympy.Expr) -> sympy.Expr:
    """Run the expensive SymPy simplification pipeline (called in a thread)."""
    # Only run nsimplify if there are floating-point atoms — it's expensive otherwise
    has_floats = any(isinstance(a, sympy.Float) for a in expr.atoms())
    if has_floats:
        result = sympy.nsimplify(expr, rational=False, tolerance=1e-8)
    else:
        result = expr

    # Cheap structural cleanup: cancel common factors, rationalize denominators
    result = sympy.cancel(sympy.radsimp(result))

    op_count = sympy.count_ops(result)

    # Fast-exit: already trivial
    if op_count <= 3:
        return result

    # Medium complexity: use expand + factor (cheaper than simplify)
    if op_count <= 30:
        candidate = sympy.factor(sympy.expand(result))
        if sympy.count_ops(candidate) < op_count:
            return candidate
        return result

    # Heavy path: restrict simplify to algebraic strategies only (no trig, no special funcs)
    result = sympy.simplify(result, ratio=1.0, measure=sympy.count_ops)
    return result


def _simplify_with_timeout(expr: sympy.Expr) -> sympy.Expr | None:
    """Run _do_sympy_simplify with a wall-clock timeout. Returns None on timeout.

    IMPORTANT: do NOT use 'with ThreadPoolExecutor() as ex' here — the context
    manager calls shutdown(wait=True) on __exit__, which blocks until the SymPy
    thread finishes and completely defeats the timeout.
    """
    future = _SIMPLIFY_EXECUTOR.submit(_do_sympy_simplify, expr)
    try:
        return future.result(timeout=_SIMPLIFY_TIMEOUT)
    except concurrent.futures.TimeoutError:
        logger.debug("SymPy simplification timed out, skipping individual")
        future.cancel()  # no-op if already running, but signals intent
        return None
    except Exception:
        return None


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
    # Fix 4: skip individuals we already attempted to simplify (and failed)
    if getattr(individual, "_simplify_attempted", False):
        return False

    original_len = len(individual)
    if original_len <= 3:
        return False

    sym_expr = _deap_to_sympy(individual, n_features)
    if sym_expr is None:
        individual._simplify_attempted = True  # type: ignore[attr-defined]
        return False

    # Fix 1 + 2: cheaper pipeline, with timeout guard
    simplified = _simplify_with_timeout(sym_expr)
    if simplified is None:
        individual._simplify_attempted = True  # type: ignore[attr-defined]
        return False

    new_tokens = _sympy_to_deap_tokens(simplified, pset, n_features)
    if new_tokens is None:
        individual._simplify_attempted = True  # type: ignore[attr-defined]
        return False

    # only accept if shorter
    if len(new_tokens) >= original_len:
        individual._simplify_attempted = True  # type: ignore[attr-defined]
        return False

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
            individual._simplify_attempted = True  # type: ignore[attr-defined]
            return False

        # Only accept if the simplified tree does not significantly degrade MSE
        if new_fitness[0] > individual.fitness.values[0] + 1e-6:
            individual._simplify_attempted = True  # type: ignore[attr-defined]
            return False
    except Exception:
        individual._simplify_attempted = True  # type: ignore[attr-defined]
        return False

    # replace contents in-place; reset the flag so the new (shorter) tree can be re-tried
    individual[0 : len(individual)] = new_tokens
    individual.fitness.values = new_fitness  # type: ignore[attr-defined]
    individual._simplify_attempted = False  # type: ignore[attr-defined]
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
