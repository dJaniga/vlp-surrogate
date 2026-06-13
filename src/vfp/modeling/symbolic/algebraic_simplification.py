from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
import sympy
from deap import creator, gp

from .helpers import evaluate_individual, invalidate_individual_caches, make_constant_terminal
from .runtime_options import DEFAULT_RUNTIME_OPTIONS, RuntimeOptions

logger = logging.getLogger(__name__)

_PENALTY_FITNESS = (1e18, 1e18)

_SIMPLIFY_TIMEOUT: float = 0.25
_SIMPLIFY_PASS_BUDGET_SECONDS: float = 2.0
_INLINE_OP_THRESHOLD: int = 40
_MAX_SIMPLIFY_TREE_SIZE: int = 80
_MAX_SIMPLIFY_OPS: int = 120

_SYMPY_SYMBOLS: dict[int, list[sympy.Symbol]] = {}
_SYM_NAME_MAP: dict[int, dict[str, str]] = {}
_SYMPY_OP_MAP: dict[str, Callable[..., sympy.Expr]] | None = None
_PSET_LOOKUP_CACHE: dict[
    int, tuple[dict[str, gp.Primitive], dict[str, gp.Terminal]]
] = {}
_RESULT_CACHE: dict[tuple, list[gp.Primitive | gp.Terminal] | None] = {}
_RESULT_CACHE_MAX = 50_000

_Float = sympy.Float
_count_ops = sympy.count_ops


def shutdown_simplification_executor() -> None:
    return None


def clear_simplification_caches() -> None:
    global _SYMPY_OP_MAP
    _SYMPY_SYMBOLS.clear()
    _SYM_NAME_MAP.clear()
    _PSET_LOOKUP_CACHE.clear()
    _RESULT_CACHE.clear()
    _SYMPY_OP_MAP = None


def _get_sympy_symbols(n: int, runtime: RuntimeOptions) -> list[sympy.Symbol]:
    if not runtime.static_cache_enabled:
        return [sympy.Symbol(f"ARG{i}") for i in range(n)]

    syms = _SYMPY_SYMBOLS.get(n)
    if syms is None:
        syms = [sympy.Symbol(f"ARG{i}") for i in range(n)]
        _SYMPY_SYMBOLS[n] = syms
        _SYM_NAME_MAP[n] = {s.name: f"ARG{i}" for i, s in enumerate(syms)}
    return syms


def _sympy_op_map(runtime: RuntimeOptions) -> dict[str, Callable[..., sympy.Expr]]:
    global _SYMPY_OP_MAP

    if not runtime.static_cache_enabled or _SYMPY_OP_MAP is None:
        op_map = {
            "_add": lambda a, b: a + b,
            "_sub": lambda a, b: a - b,
            "_mul": lambda a, b: a * b,
            "_protected_div": lambda a, b: a / b,
            "_square": lambda a: a**2,
            "_neg": lambda a: -a,
            "_abs": lambda a: sympy.Abs(a),
            "_protected_sqrt": lambda a: sympy.sqrt(sympy.Abs(a)),
        }
        if not runtime.static_cache_enabled:
            return op_map
        _SYMPY_OP_MAP = op_map

    return _SYMPY_OP_MAP


def _get_pset_lookups(
    pset: gp.PrimitiveSet,
    runtime: RuntimeOptions,
) -> tuple[dict[str, gp.Primitive], dict[str, gp.Terminal]]:
    if not runtime.static_cache_enabled:
        return (
            {p.name: p for prims in pset.primitives.values() for p in prims},
            {t.name: t for terms in pset.terminals.values() for t in terms},
        )

    key = id(pset)
    cached = _PSET_LOOKUP_CACHE.get(key)
    if cached is None:
        cached = (
            {p.name: p for prims in pset.primitives.values() for p in prims},
            {t.name: t for terms in pset.terminals.values() for t in terms},
        )
        _PSET_LOOKUP_CACHE[key] = cached
    return cached


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
    runtime: RuntimeOptions,
) -> sympy.Expr | None:
    symbols = _get_sympy_symbols(n_features, runtime)
    op_map = _sympy_op_map(runtime)
    stack: list[sympy.Expr] = []

    for node in reversed(individual):
        if isinstance(node, gp.Terminal):
            name = node.name
            if name.startswith("ARG"):
                try:
                    stack.append(symbols[int(name[3:])])
                except (ValueError, IndexError):
                    return None
            else:
                try:
                    stack.append(sympy.Float(float(node.value)))
                except (TypeError, ValueError):
                    return None
        elif isinstance(node, gp.Primitive):
            fn = op_map.get(node.name)
            if fn is None or len(stack) < node.arity:
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
    runtime: RuntimeOptions,
) -> list[gp.Primitive | gp.Terminal] | None:
    _get_sympy_symbols(n_features, runtime)
    sym_name_map = (
        _SYM_NAME_MAP[n_features]
        if runtime.static_cache_enabled
        else {f"ARG{i}": f"ARG{i}" for i in range(n_features)}
    )
    prim_by_name, term_by_name = _get_pset_lookups(pset, runtime)
    tokens: list[gp.Primitive | gp.Terminal] = []

    def walk(e: sympy.Expr) -> bool:
        if isinstance(e, sympy.Symbol):
            name = sym_name_map.get(e.name)
            term = term_by_name.get(name) if name is not None else None
            if term is None:
                return False
            tokens.append(term)
            return True

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
            return walk(e.args[0])

        if isinstance(e, sympy.Pow):
            base_expr, exp_expr = e.args
            if exp_expr == 2:
                prim = prim_by_name.get("_square")
                if prim is None:
                    return False
                tokens.append(prim)
                return walk(base_expr)

            if exp_expr == sympy.Rational(1, 2) or exp_expr == sympy.Float(0.5):
                prim = prim_by_name.get("_protected_sqrt")
                if prim is None:
                    return False
                tokens.append(prim)
                return walk(base_expr)

            if exp_expr == -1:
                prim = prim_by_name.get("_protected_div")
                if prim is None:
                    return False
                tokens.append(prim)
                tokens.append(make_constant_terminal(1.0))
                return walk(base_expr)

            return False

        prim_name = _SYMPY_TO_DEAP_BINARY.get(type(e))
        if prim_name is not None:
            prim = prim_by_name.get(prim_name)
            if prim is None:
                return False

            args = e.args
            if len(args) < 2:
                return walk(args[0]) if len(args) == 1 else False

            for _ in range(len(args) - 1):
                tokens.append(prim)

            for sub in args[:-1]:
                if not walk(sub):
                    return False

            return walk(args[-1])

        return False

    if not walk(expr):
        return None
    return tokens


def _has_radical(expr: sympy.Expr) -> bool:
    return any(
        p.exp.is_Rational and not p.exp.is_Integer for p in expr.atoms(sympy.Pow)
    )


def _do_sympy_simplify(expr: sympy.Expr) -> sympy.Expr:
    result = expr

    if result.has(_Float):
        result = sympy.nsimplify(result, rational=False, tolerance=1e-8)

    if _has_radical(result):
        result = sympy.radsimp(result)

    result = sympy.cancel(result)
    op_count = _count_ops(result)

    if op_count <= 3:
        return result

    if op_count <= 30:
        candidate = sympy.factor(sympy.expand(result))
        return candidate if _count_ops(candidate) < op_count else result

    candidate = sympy.factor(result)
    return candidate if _count_ops(candidate) < op_count else result


def _simplify_expr(expr: sympy.Expr) -> sympy.Expr | None:
    try:
        op_count = _count_ops(expr)
    except Exception:
        return None

    if op_count > _MAX_SIMPLIFY_OPS:
        return None

    start = time.monotonic()

    try:
        if op_count <= 3:
            result = sympy.cancel(expr)
        elif op_count <= _INLINE_OP_THRESHOLD:
            result = _do_sympy_simplify(expr)
        else:
            result = sympy.cancel(expr)

        if time.monotonic() - start > _SIMPLIFY_TIMEOUT:
            return None

        return result

    except Exception:
        return None


def _compute_simplified_tokens(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    n_features: int,
    runtime: RuntimeOptions,
) -> list[gp.Primitive | gp.Terminal] | None:
    if len(individual) > _MAX_SIMPLIFY_TREE_SIZE:
        return None

    sym_expr = _deap_to_sympy(individual, n_features, runtime)
    if sym_expr is None:
        return None

    simplified = _simplify_expr(sym_expr)
    if simplified is None:
        return None

    new_tokens = _sympy_to_deap_tokens(simplified, pset, n_features, runtime)
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
    runtime: RuntimeOptions,
) -> bool:
    original_len = len(individual)
    if original_len <= 3 or original_len > _MAX_SIMPLIFY_TREE_SIZE:
        return False

    if runtime.result_cache_enabled:
        key = _structural_key(individual)
        if key in _RESULT_CACHE:
            new_tokens = _RESULT_CACHE[key]
            if new_tokens is None:
                return False
        else:
            new_tokens = _compute_simplified_tokens(individual, pset, n_features, runtime)
            if len(_RESULT_CACHE) < _RESULT_CACHE_MAX:
                _RESULT_CACHE[key] = new_tokens
            if new_tokens is None:
                return False
    else:
        new_tokens = _compute_simplified_tokens(individual, pset, n_features, runtime)
        if new_tokens is None:
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
            runtime=runtime,
        )

        if new_fitness[0] >= _PENALTY_FITNESS[0]:
            return False

        if new_fitness[0] > individual.fitness.values[0] + 1e-6:  # type: ignore[attr-defined]
            return False

    except Exception:
        return False

    individual[0 : len(individual)] = new_tokens
    invalidate_individual_caches(individual)
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
    runtime: RuntimeOptions = DEFAULT_RUNTIME_OPTIONS,
    *,
    min_tree_size: int = 8,
    pass_budget_seconds: float = _SIMPLIFY_PASS_BUDGET_SECONDS,
) -> None:
    simplified_count = 0
    started = time.monotonic()

    for individual in island:
        if time.monotonic() - started > pass_budget_seconds:
            logger.debug(
                "Simplification pass budget exhausted",
                extra={"budget_seconds": pass_budget_seconds, "total": len(island)},
            )
            break

        if len(individual) < min_tree_size:
            continue

        if _simplify_individual(
            individual,
            pset,
            features,
            targets,
            n_features,
            parsimony_coefficient,
            max_tree_height,
            runtime,
        ):
            simplified_count += 1

    if simplified_count > 0:
        logger.debug(
            "Island simplification pass",
            extra={"simplified": simplified_count, "total": len(island)},
        )