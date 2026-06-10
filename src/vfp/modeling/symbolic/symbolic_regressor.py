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
    migrate,
    optimize_constants,
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
    key: tuple,
    runtime: RuntimeOptions,
) -> tuple[float, float] | None:
    if not runtime.fitness_cache_enabled:
        return None

    val = _fitness_cache.get(key)
    if val is not None:
        _fitness_cache.move_to_end(key)
    return val


def _fitness_cache_put(
    key: tuple,
    val: tuple[float, float],
    runtime: RuntimeOptions,
) -> None:
    if not runtime.fitness_cache_enabled:
        return

    if key in _fitness_cache:
        _fitness_cache.move_to_end(key)
        return

    if len(_fitness_cache) >= _FITNESS_CACHE_MAX:
        logger.info("LRU fitness cache full, evicting oldest quarter")
        evict = _FITNESS_CACHE_MAX // 4
        for _ in range(evict):
            _fitness_cache.popitem(last=False)

    _fitness_cache[key] = val


def _get_tree_key(
    ind: gp.PrimitiveTree,
    runtime: RuntimeOptions,
) -> tuple:
    if runtime.static_cache_enabled:
        key = getattr(ind, "_tree_key_cache", None)
        if key is not None:
            return key

    out: list = []
    for node in ind:
        if isinstance(node, gp.Terminal):
            out.append(("T", node.name, node.value))
        else:
            out.append(("P", node.name))

    key = tuple(out)

    if runtime.static_cache_enabled:
        try:
            ind._tree_key_cache = key  # type: ignore[attr-defined]
        except AttributeError:
            pass

    return key


def _invalidate_tree_key(ind: gp.PrimitiveTree) -> None:
    try:
        del ind._tree_key_cache  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _shallow_clone(ind: gp.PrimitiveTree) -> gp.PrimitiveTree:
    new = ind.__class__(list(ind))

    if ind.fitness.valid:  # type: ignore[attr-defined]
        new.fitness.values = ind.fitness.values  # type: ignore[attr-defined]

    cached_key = getattr(ind, "_tree_key_cache", None)
    if cached_key is not None:
        try:
            new._tree_key_cache = cached_key  # type: ignore[attr-defined]
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
                (ind.fitness.values[0] - min_e) / e_range  # type: ignore[attr-defined]
                + (ind.fitness.values[1] - min_c) / c_range  # type: ignore[attr-defined]
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
                (ind.fitness.values[0] - min_e) / e_range  # type: ignore[attr-defined]
                + (ind.fitness.values[1] - min_c) / c_range  # type: ignore[attr-defined]
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
    ) = args

    toolbox = build_toolbox(
        pset,
        max_tree_height=max_tree_height,
        tournament_size=tournament_size,
    )
    rng = np.random.default_rng(rng_seed)
    local_cache: dict[tuple, tuple[float, float]] = {}

    def eval_cached(ind: gp.PrimitiveTree) -> tuple[float, float]:
        if runtime.fitness_cache_enabled:
            key = (run_id, _get_tree_key(ind, runtime))
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
            local_cache[key] = fit
            return fit

        return evaluate_individual(
            ind,
            pset,
            features_scaled,
            targets_scaled,
            parsimony_coefficient,
            max_tree_height,
            runtime=runtime,
        )

    offspring = toolbox.select(island, len(island))
    offspring = [_shallow_clone(ind) for ind in offspring]

    for c1, c2 in zip(offspring[::2], offspring[1::2], strict=False):
        if rng.random() < crossover_rate:
            toolbox.mate(c1, c2)
            for child in (c1, c2):
                if child.fitness.valid:
                    del child.fitness.values
                _invalidate_tree_key(child)

    for mutant in offspring:
        if rng.random() >= mutation_rate:
            continue

        try:
            toolbox.mutate(mutant)
            if mutant.fitness.valid:
                del mutant.fitness.values
            _invalidate_tree_key(mutant)
        except Exception:
            pass

    for child in offspring:
        if not child.fitness.valid:
            child.fitness.values = eval_cached(child)

    survivors = tools.selNSGA2(island + offspring, island_size)

    if const_opt_top_k_ratio > 0:
        top_k = max(1, int(island_size * const_opt_top_k_ratio))
        elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]

        for ind in elite:
            optimize_constants(
                ind,
                pset,
                features_scaled,
                targets_scaled,
                timeout=const_opt_timeout,
            )
            _invalidate_tree_key(ind)
            ind.fitness.values = eval_cached(ind)

    return survivors, local_cache


@dataclass(slots=True)
class SymbolicRegressor(VFPModel):
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.3
    crossover_rate: float = 0.9
    tournament_size: int = 10
    max_tree_height: int = 6
    parsimony_coefficient: float = 1e-4
    tolerance: float = 1e-4
    seed: int | None = None
    n_islands: int = 10
    migration_interval: int = 10
    migration_size: int = 10
    algebraic_simplification: bool = True
    simplify_interval: int = 50
    basic_arithmetic_only: bool = False
    const_opt_top_k_ratio: float = 0.10
    const_opt_interval: int = 5
    parallel_islands: bool = True
    cache_mode: CacheMode = DEFAULT_RUNTIME_OPTIONS.cache_mode
    execution_mode: ExecutionMode = DEFAULT_RUNTIME_OPTIONS.execution_mode
    scale: bool = False
    max_eval_time_seconds: float = 1800.0
    pareto_front_: list[gp.PrimitiveTree] = field(default_factory=list)
    best_individual_: gp.PrimitiveTree | None = None
    _toolbox: base.Toolbox | None = None
    _pset: gp.PrimitiveSet | None = None
    _feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    _target_scaler: StandardScaler = field(default_factory=StandardScaler)
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)
    _last_features_id: tuple | None = field(default=None, repr=False)
    _built_features_name: tuple[str, ...] | None = field(default=None, repr=False)

    def __str__(self) -> str:
        return "symbolic_regressor"

    @property
    def runtime(self) -> RuntimeOptions:
        execution_mode = self.execution_mode
        if execution_mode == "auto" and not self.parallel_islands:
            execution_mode = "sequential"
        return RuntimeOptions(cache_mode=self.cache_mode, execution_mode=execution_mode)

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
            raise ValueError("execution_mode must be one of: auto, sequential, threaded.")

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

    def _evaluate_with_cache(
        self,
        ind: gp.PrimitiveTree,
        features: np.ndarray,
        targets: np.ndarray,
        runtime: RuntimeOptions,
        run_id: str,
    ) -> tuple[float, float]:
        if runtime.fitness_cache_enabled:
            key = (run_id, _get_tree_key(ind, runtime))
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

    def _evolve_one_island(
        self,
        island: list[gp.PrimitiveTree],
        island_size: int,
        features_scaled: np.ndarray,
        targets_scaled: np.ndarray,
        rng: np.random.Generator,
        runtime: RuntimeOptions,
        run_id: str,
        *,
        run_const_opt: bool,
    ) -> list[gp.PrimitiveTree]:
        toolbox = self._toolbox
        assert toolbox is not None

        offspring = toolbox.select(island, len(island))  # type: ignore[attr-defined]
        offspring = [_shallow_clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2], strict=False):
            if rng.random() < self.crossover_rate:
                toolbox.mate(c1, c2)  # type: ignore[attr-defined]
                for child in (c1, c2):
                    if child.fitness.valid:  # type: ignore[attr-defined]
                        del child.fitness.values  # type: ignore[attr-defined]
                    _invalidate_tree_key(child)

        for mutant in offspring:
            if rng.random() >= self.mutation_rate:
                continue

            try:
                toolbox.mutate(mutant)
                if mutant.fitness.valid:
                    del mutant.fitness.values
                _invalidate_tree_key(mutant)
            except Exception:
                logger.exception("Error mutating individual")

        for child in offspring:
            if not child.fitness.valid:  # type: ignore[attr-defined]
                child.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    child,
                    features_scaled,
                    targets_scaled,
                    runtime,
                    run_id,
                )

        survivors = tools.selNSGA2(island + offspring, island_size)

        if run_const_opt and self.const_opt_top_k_ratio > 0:
            top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
            elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]

            for ind in elite:
                optimize_constants(
                    ind,
                    self._pset,
                    features_scaled,
                    targets_scaled,
                    timeout=_CONST_OPT_TIMEOUT,
                )
                _invalidate_tree_key(ind)
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind,
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

        runtime = self.runtime
        run_id = uuid.uuid4().hex

        if runtime.force_sequential:
            self.close()

        saved_random_state = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)

        try:
            return self._fit_impl(
                features,
                targets,
                features_name,
                eval_set,
                runtime,
                run_id,
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
        rng = np.random.default_rng(self.seed)
        n_features = features.shape[1]

        new_feature_names = (
            tuple(features_name)
            if features_name
            else tuple(f"ARG{i}" for i in range(n_features))
        )

        features_id = (
            features.shape,
            features.dtype,
            features.data.tobytes()[:256],
        )

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
            self._last_features_id = features_id if runtime.static_cache_enabled else None

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

        rebuild_toolbox = (
            self._pset is None
            or self._toolbox is None
            or self._built_features_name != new_feature_names
            or not runtime.static_cache_enabled
        )

        if rebuild_toolbox:
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

        logger.debug(
            "Initializing symbolic regression",
            extra={
                "population_size": self.population_size,
                "generations": self.generations,
                "n_islands": self.n_islands,
                "cache_mode": self.cache_mode,
                "execution_mode": self.execution_mode,
            },
        )

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

            for ind in island:
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind,
                    features_scaled,
                    targets_scaled,
                    runtime,
                    run_id,
                )

            if self.const_opt_top_k_ratio > 0:
                top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
                elite = sorted(island, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]

                for ind in elite:
                    optimize_constants(
                        ind,
                        self._pset,
                        features_scaled,
                        targets_scaled,
                        timeout=_CONST_OPT_TIMEOUT,
                    )
                    _invalidate_tree_key(ind)
                    ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                        ind,
                        features_scaled,
                        targets_scaled,
                        runtime,
                        run_id,
                    )

            island = tools.selNSGA2(island, len(island))
            islands.append(island)

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
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.n_islands)
            executor = self._executor

        fit_start = time.monotonic()
        fit_deadline = fit_start + self.max_eval_time_seconds
        gen_timeout_floor = 10.0

        for generation in range(1, self.generations + 1):
            now = time.monotonic()

            if now >= fit_deadline:
                logger.warning(
                    "fit() wall-clock limit reached (%.0fs); stopping after generation %d",
                    self.max_eval_time_seconds,
                    generation - 1,
                )
                break

            remaining_gens = self.generations - generation + 1
            gen_timeout = max(gen_timeout_floor, (fit_deadline - now) / remaining_gens)
            run_const_opt = generation % self.const_opt_interval == 0

            if executor is not None:
                worker_args = [
                    (
                        islands[i],
                        island_size,
                        features_scaled,
                        targets_scaled,
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
                    )
                    for i in range(self.n_islands)
                ]

                futures: list[Future] = [
                    executor.submit(_evolve_island_worker, args)
                    for args in worker_args
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
                            features_scaled,
                            targets_scaled,
                            rng,
                            runtime,
                            run_id,
                            run_const_opt=run_const_opt,
                        )
                    )

                islands = evolved

                if timed_out:
                    break

            if (
                self.algebraic_simplification
                and self.simplify_interval > 0
                and generation % self.simplify_interval == 0
            ):
                for island in islands:
                    front = tools.sortNondominated(
                        island,
                        len(island),
                        first_front_only=True,
                    )[0]
                    simplify_island(
                        front,
                        self._pset,
                        features_scaled,
                        targets_scaled,
                        n_features,
                        self.parsimony_coefficient,
                        self.max_tree_height,
                        runtime,
                    )

            if (
                self.n_islands > 1
                and self.migration_interval > 0
                and generation % self.migration_interval == 0
            ):
                migrate(islands, self.migration_size, rng)

            best = _best_by_fitness_across_islands(islands)

            if best and best.fitness.values[0] < self.tolerance:  # type: ignore[attr-defined]
                logger.debug(
                    "Early stopping reached (train tolerance)",
                    extra={
                        "generation": generation,
                        "fitness": best.fitness.values[0],  # type: ignore[attr-defined]
                    },
                )
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

                improved = val_fitness < (best_eval_fitness - 1e-8)

                if improved:
                    best_eval_fitness = val_fitness
                    patience_counter = 0
                    best_islands_snapshot = [
                        [_shallow_clone(ind) for ind in island] for island in islands
                    ]
                else:
                    patience_counter += 1

                    if patience_counter >= patience:
                        logger.debug(
                            "Early stopping reached (validation patience)",
                            extra={
                                "generation": generation,
                                "val_fitness": val_fitness,
                                "best_val_fitness": best_eval_fitness,
                                "patience_counter": patience_counter,
                                "patience": patience,
                            },
                        )
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

        if best_islands_snapshot is not None and best_islands_snapshot is not islands:
            islands = best_islands_snapshot
        else:
            islands = [[_shallow_clone(ind) for ind in island] for island in islands]

        all_individuals = [ind for island in islands for ind in island]

        if self.algebraic_simplification and self.simplify_interval > 0:
            front = tools.sortNondominated(
                all_individuals,
                len(all_individuals),
                first_front_only=True,
            )[0]
            simplify_island(
                front,
                self._pset,
                features_scaled,
                targets_scaled,
                n_features,
                self.parsimony_coefficient,
                self.max_tree_height,
                runtime,
            )

        self.pareto_front_ = tools.sortNondominated(
            all_individuals,
            len(all_individuals),
            first_front_only=True,
        )[0]

        self.best_individual_ = _best_by_fitness(self.pareto_front_)

        if self.best_individual_ is None:
            raise RuntimeError("Symbolic regression did not produce a valid model.")

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