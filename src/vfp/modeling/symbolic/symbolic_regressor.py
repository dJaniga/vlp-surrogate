from __future__ import annotations

import logging
import random
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from deap import base, gp, tools
from sklearn.preprocessing import StandardScaler

from vfp.modeling.base import VFPModel
from .algebraic_simplification import (
    clear_simplification_caches,
    shutdown_simplification_executor,
    simplify_island,
)
from .helpers import (
    _CONST_OPT_TIMEOUT,
    _FORCE_SIMPLICITY,
    build_seed_individuals,
    clear_helper_caches,
    evaluate_individual,
    has_numeric_constants,
    invalidate_individual_caches,
    migrate,
    optimize_constants,
    tree_key,
    vectorised_evaluate,
)
from .primitives import build_primitive_set
from .runtime_options import (
    DEFAULT_RUNTIME_OPTIONS,
    CacheMode,
    ExecutionMode,
    RuntimeOptions,
)
from .toolbox import build_toolbox

logger = logging.getLogger(__name__)

_PENALTY_FITNESS = (1e18, 1e18)

_FITNESS_CACHE_MAX = 100_000
_fitness_cache: OrderedDict[tuple, tuple[float, float]] = OrderedDict()


def clear_symbolic_regressor_caches() -> None:
    _fitness_cache.clear()
    clear_helper_caches()
    clear_simplification_caches()


def _fitness_cache_get(
    key: tuple, runtime: RuntimeOptions
) -> tuple[float, float] | None:
    if not runtime.fitness_cache_enabled:
        return None

    val = _fitness_cache.get(key)
    if val is not None:
        _fitness_cache.move_to_end(key)

    return val


def _fitness_cache_put(
    key: tuple, val: tuple[float, float], runtime: RuntimeOptions
) -> None:
    if not runtime.fitness_cache_enabled:
        return

    if key in _fitness_cache:
        _fitness_cache.move_to_end(key)
        return

    if len(_fitness_cache) >= _FITNESS_CACHE_MAX:
        evict = _FITNESS_CACHE_MAX // 4
        for _ in range(evict):
            _fitness_cache.popitem(last=False)

    _fitness_cache[key] = val


def _shallow_clone(ind: gp.PrimitiveTree) -> gp.PrimitiveTree:
    new = ind.__class__(list(ind))

    if ind.fitness.valid:  # type: ignore[attr-defined]
        new.fitness.values = ind.fitness.values  # type: ignore[attr-defined]

    for attr in ("_tree_key_cache", "_has_numeric_constants_cache"):
        cached = getattr(ind, attr, None)
        if cached is not None:
            try:
                setattr(new, attr, cached)
            except AttributeError:
                pass

    return new


def _best_by_fitness(population: list[gp.PrimitiveTree]) -> gp.PrimitiveTree | None:
    if not population:
        return None

    valid = [
        ind
        for ind in population
        if ind.fitness.valid and ind.fitness.values[0] < _PENALTY_FITNESS[0]  # type: ignore[attr-defined]
    ]

    if not valid:
        return min(population, key=lambda ind: ind.fitness.values[0])  # type: ignore[attr-defined]

    if _FORCE_SIMPLICITY:
        errors = [ind.fitness.values[0] for ind in valid]  # type: ignore[attr-defined]
        complexities = [ind.fitness.values[1] for ind in valid]  # type: ignore[attr-defined]
        min_e, max_e = min(errors), max(errors)
        min_c, max_c = min(complexities), max(complexities)
        e_range = max_e - min_e + 1e-12
        c_range = max_c - min_c + 1e-12

        return min(
            valid,
            key=lambda ind: (  # type: ignore[attr-defined]
                (ind.fitness.values[0] - min_e) / e_range
                + (ind.fitness.values[1] - min_c) / c_range
            ),
        )

    return min(valid, key=lambda ind: ind.fitness.values[0])  # type: ignore[attr-defined]


def _best_by_fitness_across_islands(
    islands: list[list[gp.PrimitiveTree]],
) -> gp.PrimitiveTree | None:
    if _FORCE_SIMPLICITY:
        valid = [
            ind
            for island in islands
            for ind in island
            if ind.fitness.valid and ind.fitness.values[0] < _PENALTY_FITNESS[0]  # type: ignore[attr-defined]
        ]

        if not valid:
            return None

        errors = [ind.fitness.values[0] for ind in valid]  # type: ignore[attr-defined]
        complexities = [ind.fitness.values[1] for ind in valid]  # type: ignore[attr-defined]
        min_e, max_e = min(errors), max(errors)
        min_c, max_c = min(complexities), max(complexities)
        e_range = max_e - min_e + 1e-12
        c_range = max_c - min_c + 1e-12

        return min(
            valid,
            key=lambda ind: (  # type: ignore[attr-defined]
                (ind.fitness.values[0] - min_e) / e_range
                + (ind.fitness.values[1] - min_c) / c_range
            ),
        )

    best: gp.PrimitiveTree | None = None
    best_fitness = float("inf")

    for island in islands:
        for ind in island:
            if not ind.fitness.valid:  # type: ignore[attr-defined]
                continue

            f = ind.fitness.values[0]  # type: ignore[attr-defined]
            if f < best_fitness and f < _PENALTY_FITNESS[0]:
                best_fitness = f
                best = ind

    return best


def _population_diversity_metrics(
    islands: list[list[gp.PrimitiveTree]],
    runtime: RuntimeOptions,
) -> dict[str, float]:
    population = [ind for island in islands for ind in island]
    total = len(population)

    if total == 0:
        return {
            "population_size": 0.0,
            "unique_structures": 0.0,
            "structural_diversity": 0.0,
            "duplicate_ratio": 0.0,
            "avg_tree_size": 0.0,
            "std_tree_size": 0.0,
            "min_tree_size": 0.0,
            "max_tree_size": 0.0,
            "avg_height": 0.0,
            "std_height": 0.0,
            "fitness_diversity": 0.0,
            "island_diversity_mean": 0.0,
            "island_diversity_min": 0.0,
            "island_diversity_max": 0.0,
        }

    keys = [tree_key(ind, runtime) for ind in population]
    unique_structures = len(set(keys))
    structural_diversity = unique_structures / total

    sizes = np.asarray([len(ind) for ind in population], dtype=np.float64)
    heights = np.asarray([ind.height for ind in population], dtype=np.float64)

    valid_fitness = np.asarray(
        [
            ind.fitness.values[0]  # type: ignore[attr-defined]
            for ind in population
            if ind.fitness.valid and ind.fitness.values[0] < _PENALTY_FITNESS[0]  # type: ignore[attr-defined]
        ],
        dtype=np.float64,
    )

    island_diversities: list[float] = []
    for island in islands:
        if not island:
            continue
        island_keys = {tree_key(ind, runtime) for ind in island}
        island_diversities.append(len(island_keys) / len(island))

    island_diversity_arr = np.asarray(island_diversities, dtype=np.float64)

    return {
        "population_size": float(total),
        "unique_structures": float(unique_structures),
        "structural_diversity": float(structural_diversity),
        "duplicate_ratio": float(1.0 - structural_diversity),
        "avg_tree_size": float(np.mean(sizes)),
        "std_tree_size": float(np.std(sizes)),
        "min_tree_size": float(np.min(sizes)),
        "max_tree_size": float(np.max(sizes)),
        "avg_height": float(np.mean(heights)),
        "std_height": float(np.std(heights)),
        "fitness_diversity": float(np.std(valid_fitness))
        if valid_fitness.size
        else 0.0,
        "island_diversity_mean": (
            float(np.mean(island_diversity_arr)) if island_diversity_arr.size else 0.0
        ),
        "island_diversity_min": (
            float(np.min(island_diversity_arr)) if island_diversity_arr.size else 0.0
        ),
        "island_diversity_max": (
            float(np.max(island_diversity_arr)) if island_diversity_arr.size else 0.0
        ),
    }


def _evaluate_unique(
    individuals: list[gp.PrimitiveTree],
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    parsimony_coefficient: float,
    max_tree_height: int,
    runtime: RuntimeOptions,
    run_id: str,
) -> dict[int, tuple[float, float]]:
    by_key: dict[tuple, gp.PrimitiveTree] = {}
    id_to_key: dict[int, tuple] = {}

    for ind in individuals:
        key = (run_id, tree_key(ind, runtime))
        by_key.setdefault(key, ind)
        id_to_key[id(ind)] = key

    key_to_fit: dict[tuple, tuple[float, float]] = {}

    for key, ind in by_key.items():
        cached = _fitness_cache_get(key, runtime)
        if cached is not None:
            key_to_fit[key] = cached
            continue

        fit = evaluate_individual(
            ind,
            pset,
            features,
            targets,
            parsimony_coefficient,
            max_tree_height,
            runtime=runtime,
        )
        key_to_fit[key] = fit
        _fitness_cache_put(key, fit, runtime)

    return {ind_id: key_to_fit[key] for ind_id, key in id_to_key.items()}


def _deduplicate_island_structures(
    island: list[gp.PrimitiveTree],
    toolbox: base.Toolbox,
    island_size: int,
    runtime: RuntimeOptions,
    *,
    max_attempts_multiplier: int = 20,
) -> list[gp.PrimitiveTree]:
    if not island:
        return island

    unique: list[gp.PrimitiveTree] = []
    seen: set[tuple] = set()

    for ind in sorted(
        island,
        key=lambda candidate: (
            candidate.fitness.values[0]  # type: ignore[attr-defined]
            if candidate.fitness.valid  # type: ignore[attr-defined]
            else float("inf")
        ),
    ):
        key = tree_key(ind, runtime)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ind)
        if len(unique) >= island_size:
            return unique

    attempts = 0
    max_attempts = max(island_size * max_attempts_multiplier, island_size)

    while len(unique) < island_size and attempts < max_attempts:
        attempts += 1
        candidate = toolbox.individual()  # type: ignore[attr-defined]
        invalidate_individual_caches(candidate)
        key = tree_key(candidate, runtime)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    while len(unique) < island_size:
        candidate = toolbox.individual()  # type: ignore[attr-defined]
        invalidate_individual_caches(candidate)
        unique.append(candidate)

    return unique


def _replace_worst_with_random_unique(
    island: list[gp.PrimitiveTree],
    toolbox: base.Toolbox,
    replace_count: int,
    runtime: RuntimeOptions,
) -> list[gp.PrimitiveTree]:
    if replace_count <= 0 or not island:
        return island

    replace_count = min(replace_count, len(island))
    survivors = sorted(
        island,
        key=lambda candidate: (
            candidate.fitness.values[0]  # type: ignore[attr-defined]
            if candidate.fitness.valid  # type: ignore[attr-defined]
            else float("inf")
        ),
    )[: len(island) - replace_count]

    seen = {tree_key(ind, runtime) for ind in survivors}
    replacements: list[gp.PrimitiveTree] = []
    attempts = 0
    max_attempts = max(replace_count * 30, replace_count)

    while len(replacements) < replace_count and attempts < max_attempts:
        attempts += 1
        candidate = toolbox.individual()  # type: ignore[attr-defined]
        invalidate_individual_caches(candidate)
        key = tree_key(candidate, runtime)
        if key in seen:
            continue
        seen.add(key)
        replacements.append(candidate)

    while len(replacements) < replace_count:
        candidate = toolbox.individual()  # type: ignore[attr-defined]
        invalidate_individual_caches(candidate)
        replacements.append(candidate)

    return survivors + replacements


def _evolve_island_worker(args: tuple) -> tuple[list[gp.PrimitiveTree], dict]:
    (
        island,
        island_size,
        features_scaled,
        targets_scaled,
        pset,
        max_tree_height,
        tournament_size,
        parsimony_coefficient,
        crossover_rate,
        mutation_rate,
        const_opt_top_k_ratio,
        rng_seed,
        const_opt_timeout,
        runtime,
        run_id,
        const_opt_sample_size,
        const_opt_max_constants,
        const_opt_min_improvement,
    ) = args

    toolbox = build_toolbox(
        pset,
        max_tree_height=max_tree_height,
        tournament_size=tournament_size,
    )
    rng = np.random.default_rng(rng_seed)
    local_cache: dict[tuple, tuple[float, float]] = {}

    def eval_cached(ind: gp.PrimitiveTree) -> tuple[float, float]:
        key = (run_id, tree_key(ind, runtime))
        if runtime.fitness_cache_enabled:
            hit = local_cache.get(key)
            if hit is not None:
                return hit

        fit = evaluate_individual(
            ind,
            pset,
            features_scaled,
            targets_scaled,
            parsimony_coefficient,
            max_tree_height,
            runtime=runtime,
        )

        if runtime.fitness_cache_enabled:
            local_cache[key] = fit

        return fit

    offspring = toolbox.select(island, len(island))
    offspring = [_shallow_clone(ind) for ind in offspring]

    for c1, c2 in zip(offspring[::2], offspring[1::2], strict=False):
        if rng.random() < crossover_rate:
            toolbox.mate(c1, c2)
            for child in (c1, c2):
                if child.fitness.valid:
                    del child.fitness.values
                invalidate_individual_caches(child)

    for mutant in offspring:
        if rng.random() >= mutation_rate:
            continue

        try:
            toolbox.mutate(mutant)
            if mutant.fitness.valid:
                del mutant.fitness.values
            invalidate_individual_caches(mutant)
        except Exception:
            pass

    invalid = [child for child in offspring if not child.fitness.valid]
    if invalid:
        by_key: dict[tuple, gp.PrimitiveTree] = {}
        for ind in invalid:
            by_key.setdefault((run_id, tree_key(ind, runtime)), ind)

        key_to_fit: dict[tuple, tuple[float, float]] = {}
        for key, ind in by_key.items():
            hit = local_cache.get(key) if runtime.fitness_cache_enabled else None
            fit = hit if hit is not None else eval_cached(ind)
            key_to_fit[key] = fit

        for ind in invalid:
            ind.fitness.values = key_to_fit[(run_id, tree_key(ind, runtime))]

    survivors = tools.selNSGA2(island + offspring, island_size)

    if const_opt_top_k_ratio > 0:
        top_k = max(1, int(island_size * const_opt_top_k_ratio))
        elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]
        elite = [ind for ind in elite if has_numeric_constants(ind, runtime)]

        for ind in elite:
            before = ind.fitness.values[0]
            improved = optimize_constants(
                ind,
                pset,
                features_scaled,
                targets_scaled,
                timeout=const_opt_timeout,
                sample_size=const_opt_sample_size,
                max_constants=const_opt_max_constants,
            )
            if not improved:
                continue

            fit = eval_cached(ind)
            if before - fit[0] >= const_opt_min_improvement:
                ind.fitness.values = fit
            else:
                ind.fitness.values = fit

    return survivors, local_cache


@dataclass(slots=True)
class SymbolicRegressor(VFPModel):
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.3
    crossover_rate: float = 0.9
    tournament_size: int = 4
    max_tree_height: int = 8
    parsimony_coefficient: float = 1e-4
    tolerance: float = 1e-4
    seed: int | None = None
    n_islands: int = 5
    migration_interval: int = 25
    migration_size: int = 1

    algebraic_simplification: bool = True
    simplify_interval: int = 50
    simplify_top_k: int = 10
    simplify_min_tree_size: int = 8
    final_simplify_top_k: int = 20

    basic_arithmetic_only: bool = False

    const_opt_top_k_ratio: float = 0.50
    const_opt_interval: int = 5
    const_opt_sample_size: int | None = 2048
    const_opt_max_constants: int = 6
    const_opt_min_improvement: float = 1e-8
    const_opt_patience: int = 3

    early_eval_sample_ratio: float = 1.0
    full_eval_after_ratio: float = 0.7

    nsga_interval: int = 1
    parallel_islands: bool = True
    auto_disable_threading_after_generations: int = 2000

    cache_mode: CacheMode = DEFAULT_RUNTIME_OPTIONS.cache_mode
    execution_mode: ExecutionMode = DEFAULT_RUNTIME_OPTIONS.execution_mode

    log_diversity: bool = False
    print_diversity: bool = False
    diversity_log_interval: int = 1

    enforce_unique_structures: bool = True
    diversity_rescue_enabled: bool = True
    diversity_rescue_duplicate_threshold: float = 0.45
    diversity_rescue_fraction: float = 0.30
    diversity_rescue_interval: int = 1

    scale: bool = False
    max_eval_time_seconds: float = 1800.0

    pareto_front_: list[gp.PrimitiveTree] = field(default_factory=list)
    best_individual_: gp.PrimitiveTree | None = None
    profile_: dict[str, float] = field(default_factory=dict)
    diversity_history_: list[dict[str, float]] = field(default_factory=list)

    _toolbox: base.Toolbox | None = None
    _pset: gp.PrimitiveSet | None = None
    _feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    _target_scaler: StandardScaler = field(default_factory=StandardScaler)
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)
    _last_features_id: tuple | None = field(default=None, repr=False)
    _built_features_name: tuple[str, ...] | None = field(default=None, repr=False)
    _const_opt_no_improve_counter: int = field(default=0, repr=False)

    def __str__(self) -> str:
        return "symbolic_regressor"

    @property
    def runtime(self) -> RuntimeOptions:
        execution_mode = self.execution_mode
        if execution_mode == "auto" and not self.parallel_islands:
            execution_mode = "sequential"
        return RuntimeOptions(cache_mode=self.cache_mode, execution_mode=execution_mode)

    def _time_add(self, name: str, seconds: float) -> None:
        self.profile_[name] = self.profile_.get(name, 0.0) + seconds

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        shutdown_simplification_executor()

    def get_fit_details(self) -> dict[str, Any]:
        if self.best_individual_ is None:
            raise ValueError("Model has not been fit yet.")

        return {
            "pareto_size": len(self.pareto_front_),
            "best_fitness": self.best_individual_.fitness.values[0],  # type: ignore[attr-defined]
            "best_complexity": self.best_individual_.fitness.values[1],  # type: ignore[attr-defined]
            "expression": str(self.best_individual_),
            "cache_mode": self.cache_mode,
            "execution_mode": self.execution_mode,
            "profile": dict(self.profile_),
            "last_diversity": self.diversity_history_[-1]
            if self.diversity_history_
            else None,
        }

    def _validate_config(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.generations <= 0:
            raise ValueError("generations must be positive.")
        if self.n_islands <= 0:
            raise ValueError("n_islands must be positive.")
        if self.migration_interval < 0:
            raise ValueError("migration_interval must be non-negative.")
        if self.migration_size < 0:
            raise ValueError("migration_size must be non-negative.")
        if self.simplify_interval < 0:
            raise ValueError("simplify_interval must be non-negative.")
        if self.const_opt_interval <= 0:
            raise ValueError("const_opt_interval must be positive.")
        if not 0 <= self.const_opt_top_k_ratio <= 1:
            raise ValueError("const_opt_top_k_ratio must be in [0, 1].")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1].")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1].")
        if self.max_tree_height <= 0:
            raise ValueError("max_tree_height must be positive.")
        if self.max_eval_time_seconds <= 0:
            raise ValueError("max_eval_time_seconds must be positive.")
        if self.cache_mode not in {"off", "safe", "all"}:
            raise ValueError("cache_mode must be one of: off, safe, all.")
        if self.execution_mode not in {"auto", "sequential", "threaded"}:
            raise ValueError(
                "execution_mode must be one of: auto, sequential, threaded."
            )
        if not 0 < self.full_eval_after_ratio <= 1:
            raise ValueError("full_eval_after_ratio must be in (0, 1].")
        if not 0 < self.early_eval_sample_ratio <= 1:
            raise ValueError("early_eval_sample_ratio must be in (0, 1].")
        if self.diversity_log_interval <= 0:
            raise ValueError("diversity_log_interval must be positive.")
        if not 0 <= self.diversity_rescue_duplicate_threshold <= 1:
            raise ValueError("diversity_rescue_duplicate_threshold must be in [0, 1].")
        if not 0 <= self.diversity_rescue_fraction <= 1:
            raise ValueError("diversity_rescue_fraction must be in [0, 1].")
        if self.diversity_rescue_interval <= 0:
            raise ValueError("diversity_rescue_interval must be positive.")

    def _log_population_diversity(
        self,
        generation: int,
        islands: list[list[gp.PrimitiveTree]],
        runtime: RuntimeOptions,
    ) -> dict[str, float]:
        metrics = _population_diversity_metrics(islands, runtime)
        metrics["generation"] = float(generation)
        self.diversity_history_.append(metrics)

        if self.log_diversity:
            logger.debug(
                "Population diversity",
                extra={
                    "generation": generation,
                    "structural_diversity": metrics["structural_diversity"],
                    "unique_structures": int(metrics["unique_structures"]),
                    "population_size": int(metrics["population_size"]),
                    "duplicate_ratio": metrics["duplicate_ratio"],
                    "avg_tree_size": metrics["avg_tree_size"],
                    "std_tree_size": metrics["std_tree_size"],
                    "min_tree_size": metrics["min_tree_size"],
                    "max_tree_size": metrics["max_tree_size"],
                    "avg_height": metrics["avg_height"],
                    "std_height": metrics["std_height"],
                    "fitness_diversity": metrics["fitness_diversity"],
                    "island_diversity_mean": metrics["island_diversity_mean"],
                    "island_diversity_min": metrics["island_diversity_min"],
                    "island_diversity_max": metrics["island_diversity_max"],
                },
            )
        return metrics

    def _scale_features(self, features: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if not self.scale:
            return features
        return (
            self._feature_scaler.fit_transform(features)
            if fit
            else self._feature_scaler.transform(features)
        )

    def _scale_targets(self, targets: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if not self.scale:
            return np.asarray(targets).flatten()

        arr = np.asarray(targets).flatten().reshape(-1, 1)
        return (
            self._target_scaler.fit_transform(arr)
            if fit
            else self._target_scaler.transform(arr)
        ).ravel()

    def _unscale_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if not self.scale:
            return predictions
        return self._target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()

    def _fit_features_id(self, features: np.ndarray) -> tuple:
        flat = np.ascontiguousarray(features).ravel()
        head = flat[: min(32, flat.size)].tobytes()
        return (id(features), features.shape, features.dtype, head)

    def _evaluation_data_for_generation(
        self,
        generation: int,
        features_scaled: np.ndarray,
        targets_scaled: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.early_eval_sample_ratio >= 1.0
            or generation / max(self.generations, 1) >= self.full_eval_after_ratio
        ):
            return features_scaled, targets_scaled

        sample_size = max(16, int(len(targets_scaled) * self.early_eval_sample_ratio))
        sample_size = min(sample_size, len(targets_scaled))
        idx = rng.choice(len(targets_scaled), size=sample_size, replace=False)
        return features_scaled[idx], targets_scaled[idx]

    def _evaluate_with_cache(
        self,
        ind: gp.PrimitiveTree,
        features: np.ndarray,
        targets: np.ndarray,
        runtime: RuntimeOptions,
        run_id: str,
    ) -> tuple[float, float]:
        if runtime.fitness_cache_enabled:
            key = (run_id, tree_key(ind, runtime))
            cached = _fitness_cache_get(key, runtime)
            if cached is not None:
                return cached

            fit = evaluate_individual(
                ind,
                self._pset,
                features,
                targets,
                self.parsimony_coefficient,
                self.max_tree_height,
                runtime=runtime,
            )
            _fitness_cache_put(key, fit, runtime)
            return fit

        return evaluate_individual(
            ind,
            self._pset,
            features,
            targets,
            self.parsimony_coefficient,
            self.max_tree_height,
            runtime=runtime,
        )

    def _evaluate_invalid_batch(
        self,
        individuals: list[gp.PrimitiveTree],
        features: np.ndarray,
        targets: np.ndarray,
        runtime: RuntimeOptions,
        run_id: str,
    ) -> None:
        invalid = [ind for ind in individuals if not ind.fitness.valid]  # type: ignore[attr-defined]
        if not invalid:
            return

        assert self._pset is not None
        fits = _evaluate_unique(
            invalid,
            self._pset,
            features,
            targets,
            self.parsimony_coefficient,
            self.max_tree_height,
            runtime,
            run_id,
        )

        for ind in invalid:
            ind.fitness.values = fits[id(ind)]

    def _deduplicate_and_evaluate_island(
        self,
        island: list[gp.PrimitiveTree],
        island_size: int,
        features: np.ndarray,
        targets: np.ndarray,
        runtime: RuntimeOptions,
        run_id: str,
    ) -> list[gp.PrimitiveTree]:
        if not self.enforce_unique_structures:
            return island

        assert self._toolbox is not None

        deduplicated = _deduplicate_island_structures(
            island,
            self._toolbox,
            island_size,
            runtime,
        )
        self._evaluate_invalid_batch(deduplicated, features, targets, runtime, run_id)
        return deduplicated

    def _rescue_population_diversity(
        self,
        islands: list[list[gp.PrimitiveTree]],
        island_size: int,
        features: np.ndarray,
        targets: np.ndarray,
        runtime: RuntimeOptions,
        run_id: str,
        generation: int,
        diversity_metrics: dict[str, float] | None,
    ) -> list[list[gp.PrimitiveTree]]:
        if not self.diversity_rescue_enabled:
            return islands

        if generation % self.diversity_rescue_interval != 0:
            return islands

        if diversity_metrics is None:
            diversity_metrics = _population_diversity_metrics(islands, runtime)

        if (
            diversity_metrics["duplicate_ratio"]
            < self.diversity_rescue_duplicate_threshold
        ):
            return islands

        assert self._toolbox is not None

        replace_count = max(1, int(island_size * self.diversity_rescue_fraction))
        rescued: list[list[gp.PrimitiveTree]] = []

        for island in islands:
            new_island = _replace_worst_with_random_unique(
                island,
                self._toolbox,
                replace_count,
                runtime,
            )
            self._evaluate_invalid_batch(new_island, features, targets, runtime, run_id)

            if self.enforce_unique_structures:
                new_island = _deduplicate_island_structures(
                    new_island,
                    self._toolbox,
                    island_size,
                    runtime,
                )
                self._evaluate_invalid_batch(
                    new_island, features, targets, runtime, run_id
                )

            rescued.append(tools.selNSGA2(new_island, len(new_island)))

        if self.log_diversity:
            logger.debug(
                "Diversity rescue applied",
                extra={
                    "generation": generation,
                    "duplicate_ratio": diversity_metrics["duplicate_ratio"],
                    "replace_count_per_island": replace_count,
                    "islands": len(islands),
                },
            )

        return rescued

    def _run_const_opt(
        self,
        candidates: list[gp.PrimitiveTree],
        pset: gp.PrimitiveSet,
        features: np.ndarray,
        targets: np.ndarray,
        runtime: RuntimeOptions,
        run_id: str,
    ) -> None:
        if self._const_opt_no_improve_counter >= self.const_opt_patience:
            return

        any_significant_improvement = False

        for ind in candidates:
            if not has_numeric_constants(ind, runtime):
                continue

            before = ind.fitness.values[0]  # type: ignore[attr-defined]
            improved = optimize_constants(
                ind,
                pset,
                features,
                targets,
                timeout=_CONST_OPT_TIMEOUT,
                sample_size=self.const_opt_sample_size,
                max_constants=self.const_opt_max_constants,
            )
            if not improved:
                continue

            fit = self._evaluate_with_cache(ind, features, targets, runtime, run_id)
            ind.fitness.values = fit  # type: ignore[attr-defined]

            if before - fit[0] >= self.const_opt_min_improvement:
                any_significant_improvement = True

        if any_significant_improvement:
            self._const_opt_no_improve_counter = 0
        else:
            self._const_opt_no_improve_counter += 1

    def _evolve_one_island(
        self,
        island: list[gp.PrimitiveTree],
        island_size: int,
        features_scaled: np.ndarray,
        targets_scaled: np.ndarray,
        rng: np.random.Generator,
        runtime: RuntimeOptions,
        run_id: str,
        generation: int,
        *,
        run_const_opt: bool,
    ) -> list[gp.PrimitiveTree]:
        toolbox = self._toolbox
        assert toolbox is not None
        assert self._pset is not None

        offspring = toolbox.select(island, len(island))  # type: ignore[attr-defined]
        offspring = [_shallow_clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2], strict=False):
            if rng.random() < self.crossover_rate:
                toolbox.mate(c1, c2)  # type: ignore[attr-defined]
                for child in (c1, c2):
                    if child.fitness.valid:  # type: ignore[attr-defined]
                        del child.fitness.values  # type: ignore[attr-defined]
                    invalidate_individual_caches(child)

        for mutant in offspring:
            if rng.random() >= self.mutation_rate:
                continue

            try:
                toolbox.mutate(mutant)
                if mutant.fitness.valid:
                    del mutant.fitness.values
                invalidate_individual_caches(mutant)
            except Exception:
                logger.debug("Error mutating individual", exc_info=True)

        self._evaluate_invalid_batch(
            offspring, features_scaled, targets_scaled, runtime, run_id
        )

        if self.nsga_interval <= 1 or generation % self.nsga_interval == 0:
            survivors = tools.selNSGA2(island + offspring, island_size)
        else:
            survivors = tools.selBest(island + offspring, island_size)

        if run_const_opt and self.const_opt_top_k_ratio > 0:
            top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
            elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
            elite = [ind for ind in elite if has_numeric_constants(ind, runtime)]
            self._run_const_opt(
                elite,
                self._pset,
                features_scaled,
                targets_scaled,
                runtime,
                run_id,
            )

        survivors = self._deduplicate_and_evaluate_island(
            survivors,
            island_size,
            features_scaled,
            targets_scaled,
            runtime,
            run_id,
        )

        return survivors

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> SymbolicRegressor:
        self._validate_config()
        self.profile_.clear()
        self.diversity_history_.clear()
        self._const_opt_no_improve_counter = 0

        runtime = self.runtime
        run_id = uuid.uuid4().hex

        if runtime.force_sequential:
            self.close()

        saved_random_state = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)

        try:
            return self._fit_impl(
                features, targets, features_name, eval_set, runtime, run_id
            )
        finally:
            random.setstate(saved_random_state)
            _fitness_cache.clear()
            if not runtime.result_cache_enabled:
                clear_symbolic_regressor_caches()

    def _fit_impl(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None,
        eval_set: tuple[np.ndarray, np.ndarray] | None,
        runtime: RuntimeOptions,
        run_id: str,
    ) -> SymbolicRegressor:
        total_start = time.monotonic()
        rng = np.random.default_rng(self.seed)
        n_features = features.shape[1]

        new_feature_names = (
            tuple(features_name)
            if features_name
            else tuple(f"ARG{i}" for i in range(n_features))
        )

        scale_start = time.monotonic()
        features_id = self._fit_features_id(features)

        if runtime.static_cache_enabled and self._last_features_id == features_id:
            features_scaled = np.ascontiguousarray(
                self._scale_features(features, fit=False),
                dtype=np.float64,
            )
            targets_scaled = np.ascontiguousarray(
                self._scale_targets(targets, fit=False),
                dtype=np.float64,
            )
        else:
            features_scaled = np.ascontiguousarray(
                self._scale_features(features, fit=True),
                dtype=np.float64,
            )
            targets_scaled = np.ascontiguousarray(
                self._scale_targets(targets, fit=True),
                dtype=np.float64,
            )
            self._last_features_id = (
                features_id if runtime.static_cache_enabled else None
            )

        self._time_add("scale", time.monotonic() - scale_start)

        eval_features_scaled = eval_targets_scaled = None
        if eval_set is not None:
            eval_features_scaled = np.ascontiguousarray(
                self._scale_features(eval_set[0], fit=False),
                dtype=np.float64,
            )
            eval_targets_scaled = np.ascontiguousarray(
                self._scale_targets(eval_set[1], fit=False),
                dtype=np.float64,
            )

        if (
            self._pset is None
            or self._toolbox is None
            or self._built_features_name != new_feature_names
            or not runtime.static_cache_enabled
        ):
            self._pset = build_primitive_set(n_features, self.basic_arithmetic_only)
            self._pset.renameArguments(
                **{f"ARG{idx}": name for idx, name in enumerate(new_feature_names)}
            )
            self._toolbox = build_toolbox(
                self._pset,
                max_tree_height=self.max_tree_height,
                tournament_size=self.tournament_size,
            )
            self._built_features_name = new_feature_names

        assert self._pset is not None
        assert self._toolbox is not None

        self.features_name = new_feature_names

        island_size = self.population_size // self.n_islands
        if island_size < 4:
            raise ValueError(
                f"population_size={self.population_size} is too small for "
                f"n_islands={self.n_islands} (need at least 4 per island)."
            )

        init_start = time.monotonic()
        seed_individuals = build_seed_individuals(
            self._pset,
            n_features,
            include_stochastic=True,
        )

        islands: list[list[gp.PrimitiveTree]] = []
        for island_idx in range(self.n_islands):
            island = self._toolbox.population(n=island_size)  # type: ignore[attr-defined]

            if island_idx == 0 and seed_individuals:
                n_inject = min(len(seed_individuals), island_size // 2)
                island[:n_inject] = [
                    _shallow_clone(s) for s in seed_individuals[:n_inject]
                ]

            self._evaluate_invalid_batch(
                island,
                features_scaled,
                targets_scaled,
                runtime,
                run_id,
            )

            if self.const_opt_top_k_ratio > 0:
                top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
                elite = sorted(island, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
                elite = [ind for ind in elite if has_numeric_constants(ind, runtime)]
                self._run_const_opt(
                    elite,
                    self._pset,
                    features_scaled,
                    targets_scaled,
                    runtime,
                    run_id,
                )

            island = self._deduplicate_and_evaluate_island(
                island,
                island_size,
                features_scaled,
                targets_scaled,
                runtime,
                run_id,
            )

            islands.append(tools.selNSGA2(island, len(island)))

        self._time_add("initial_population", time.monotonic())

        best_eval_fitness = float("inf")
        patience = max(self.generations // 5, 10)
        patience_counter = 0
        best_islands_snapshot: list[list[gp.PrimitiveTree]] | None = None
        island_rng_seeds = [
            int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)
        ]

        use_parallel_islands = (
            self.execution_mode in {"auto", "threaded"}
            and self.parallel_islands
            and not runtime.force_sequential
            and self.n_islands > 1
        )

        executor: ThreadPoolExecutor | None = None
        if use_parallel_islands:
            self._executor = self._executor or ThreadPoolExecutor(
                max_workers=self.n_islands
            )
            executor = self._executor

        fit_deadline = time.monotonic() + self.max_eval_time_seconds
        gen_timeout_floor = 30.0
        threaded_total = 0.0
        serial_total = 0.0

        for generation in range(1, self.generations + 1):
            gen_loop_start = time.monotonic()
            now = time.monotonic()

            if now >= fit_deadline:
                logger.warning(
                    "fit() wall-clock limit reached (%.0fs); stopping after generation %d",
                    self.max_eval_time_seconds,
                    generation - 1,
                )
                break

            eval_features, eval_targets = self._evaluation_data_for_generation(
                generation,
                features_scaled,
                targets_scaled,
                rng,
            )

            remaining_gens = self.generations - generation + 1
            gen_timeout = max(gen_timeout_floor, (fit_deadline - now) / remaining_gens)
            run_const_opt = generation % self.const_opt_interval == 0

            evolve_start = time.monotonic()

            if executor is not None:
                worker_args = [
                    (
                        islands[i],
                        island_size,
                        eval_features,
                        eval_targets,
                        self._pset,
                        self.max_tree_height,
                        self.tournament_size,
                        self.parsimony_coefficient,
                        self.crossover_rate,
                        self.mutation_rate,
                        self.const_opt_top_k_ratio if run_const_opt else 0.0,
                        island_rng_seeds[i],
                        _CONST_OPT_TIMEOUT,
                        runtime,
                        run_id,
                        self.const_opt_sample_size,
                        self.const_opt_max_constants,
                        self.const_opt_min_improvement,
                    )
                    for i in range(self.n_islands)
                ]

                futures: list[Future] = [
                    executor.submit(_evolve_island_worker, args) for args in worker_args
                ]
                done, not_done = wait(
                    futures,
                    timeout=gen_timeout,
                    return_when=ALL_COMPLETED,
                )

                if not_done:
                    logger.warning(
                        "Generation %d timed out (%.1fs per-gen budget); stopping.",
                        generation,
                        gen_timeout,
                    )
                    for future in not_done:
                        future.cancel()
                    break

                results = [future.result() for future in done]
                islands = [r[0] for r in results]

                if runtime.fitness_cache_enabled:
                    for _, worker_cache in results:
                        for k, v in worker_cache.items():
                            _fitness_cache_put(k, v, runtime)

                island_rng_seeds = [
                    int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)
                ]
                threaded_total += time.monotonic() - evolve_start

                if (
                    self.execution_mode == "auto"
                    and generation >= self.auto_disable_threading_after_generations
                    and serial_total > 0
                    and threaded_total > serial_total * 1.15
                ):
                    executor = None

            else:
                gen_start = time.monotonic()
                evolved: list[list[gp.PrimitiveTree]] = []
                timed_out = False

                for i in range(self.n_islands):
                    if time.monotonic() - gen_start > gen_timeout:
                        logger.warning(
                            "Serial generation %d timed out (%.1fs); keeping remaining islands.",
                            generation,
                            gen_timeout,
                        )
                        evolved.extend(islands[i:])
                        timed_out = True
                        break

                    evolved.append(
                        self._evolve_one_island(
                            islands[i],
                            island_size,
                            eval_features,
                            eval_targets,
                            rng,
                            runtime,
                            run_id,
                            generation,
                            run_const_opt=run_const_opt,
                        )
                    )

                islands = evolved
                serial_total += time.monotonic() - evolve_start

                if timed_out:
                    break

            self._time_add("evolution", time.monotonic() - evolve_start)

            if (
                self.algebraic_simplification
                and self.simplify_interval > 0
                and generation % self.simplify_interval == 0
            ):
                simplify_start = time.monotonic()
                for island in islands:
                    candidates = sorted(
                        island,
                        key=lambda ind: ind.fitness.values[0],  # type: ignore[attr-defined]
                    )[: min(self.simplify_top_k, len(island))]
                    candidates = [
                        ind
                        for ind in candidates
                        if len(ind) >= self.simplify_min_tree_size
                    ]
                    simplify_island(
                        candidates,
                        self._pset,
                        features_scaled,
                        targets_scaled,
                        n_features,
                        self.parsimony_coefficient,
                        self.max_tree_height,
                        runtime,
                        min_tree_size=self.simplify_min_tree_size,
                        pass_budget_seconds=1.0,
                    )
                self._time_add("simplification", time.monotonic() - simplify_start)

            if (
                self.n_islands > 1
                and self.migration_interval > 0
                and generation % self.migration_interval == 0
            ):
                migration_start = time.monotonic()
                migrate(islands, self.migration_size, rng)
                self._time_add("migration", time.monotonic() - migration_start)

            best = _best_by_fitness_across_islands(islands)

            if (
                self.log_diversity or self.print_diversity
            ) and generation % self.diversity_log_interval == 0:
                diversity_start = time.monotonic()
                diversity_metrics = self._log_population_diversity(
                    generation,
                    islands,
                    runtime,
                )
                self._time_add("diversity", time.monotonic() - diversity_start)
            else:
                diversity_metrics = None

            islands = self._rescue_population_diversity(
                islands,
                island_size,
                eval_features,
                eval_targets,
                runtime,
                run_id,
                generation,
                diversity_metrics,
            )

            if (
                self.log_diversity or self.print_diversity
            ) and generation % self.diversity_log_interval == 0:
                post_rescue_metrics = _population_diversity_metrics(islands, runtime)
                diversity_metrics = post_rescue_metrics
                logger.debug(
                    "Population diversity after rescue",
                    extra={
                        "generation": generation,
                        "structural_diversity": post_rescue_metrics[
                            "structural_diversity"
                        ],
                        "unique_structures": int(
                            post_rescue_metrics["unique_structures"]
                        ),
                        "population_size": int(post_rescue_metrics["population_size"]),
                        "duplicate_ratio": post_rescue_metrics["duplicate_ratio"],
                    },
                )

            if best and best.fitness.values[0] < self.tolerance:  # type: ignore[attr-defined]
                best_islands_snapshot = islands
                break

            val_fitness = None

            if (
                best is not None
                and eval_features_scaled is not None
                and eval_targets_scaled is not None
            ):
                val_fitness, _ = evaluate_individual(
                    best,
                    self._pset,
                    eval_features_scaled,
                    eval_targets_scaled,
                    self.parsimony_coefficient,
                    self.max_tree_height,
                    runtime=runtime,
                )

                if val_fitness < (best_eval_fitness - 1e-8):
                    best_eval_fitness = val_fitness
                    patience_counter = 0
                    best_islands_snapshot = [
                        [_shallow_clone(ind) for ind in island] for island in islands
                    ]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            logger.debug(
                "Generation complete",
                extra={
                    "#": generation,
                    "train_fitness": best.fitness.values[0] if best else None,
                    "train_complexity": best.fitness.values[1] if best else None,
                    "best_val_fitness": best_eval_fitness,
                    "current_val_fitness": val_fitness,
                    "patience_counter": patience_counter,
                },
            )
            self._time_add("generation_loop", time.monotonic() - gen_loop_start)

        if best_islands_snapshot is not None and best_islands_snapshot is not islands:
            islands = best_islands_snapshot
        else:
            islands = [[_shallow_clone(ind) for ind in island] for island in islands]

        finalize_start = time.monotonic()
        all_individuals = [ind for island in islands for ind in island]

        if self.algebraic_simplification and self.simplify_interval > 0:
            candidates = sorted(
                all_individuals,
                key=lambda ind: ind.fitness.values[0],  # type: ignore[attr-defined]
            )[: min(self.final_simplify_top_k, len(all_individuals))]
            candidates = [
                ind for ind in candidates if len(ind) >= self.simplify_min_tree_size
            ]
            simplify_island(
                candidates,
                self._pset,
                features_scaled,
                targets_scaled,
                n_features,
                self.parsimony_coefficient,
                self.max_tree_height,
                runtime,
                min_tree_size=self.simplify_min_tree_size,
                pass_budget_seconds=1.5,
            )

        self.pareto_front_ = tools.sortNondominated(
            all_individuals,
            len(all_individuals),
            first_front_only=True,
        )[0]

        self.best_individual_ = _best_by_fitness(self.pareto_front_)
        if self.best_individual_ is None:
            raise RuntimeError("Symbolic regression did not produce a valid model.")

        self._time_add("finalize", time.monotonic() - finalize_start)
        self._time_add("total", time.monotonic() - total_start)

        logger.debug("Symbolic regression complete", extra=self.get_fit_details())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.best_individual_ is None or self._pset is None:
            raise ValueError("Model has not been fit yet.")

        features_scaled = self._scale_features(features, fit=False)
        func = gp.compile(self.best_individual_, self._pset)
        predictions_scaled = vectorised_evaluate(func, features_scaled)
        predictions = self._unscale_predictions(predictions_scaled)

        logger.debug(
            "Symbolic prediction complete",
            extra={"samples": int(features.shape[0])},
        )

        return predictions
