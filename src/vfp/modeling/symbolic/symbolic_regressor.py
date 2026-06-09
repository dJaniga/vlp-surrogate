from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from deap import base, gp, tools
from sklearn.preprocessing import StandardScaler

from vfp.modeling.base import VFPModel
from .algebraic_simplification import simplify_island
from .helpers import (
    _FORCE_SIMPLICITY,
    _CONST_OPT_TIMEOUT,
    build_seed_individuals,
    optimize_constants,
    evaluate_individual,
    migrate,
    vectorised_evaluate,
)
from .primitives import build_primitive_set
from .toolbox import build_toolbox

logger = logging.getLogger(__name__)

_PENALTY_FITNESS = (1e18, 1e18)

# ---- perf #9: module-level LRU fitness cache (shared by serial path) --------
_FITNESS_CACHE_MAX = 50_000
_fitness_cache: OrderedDict[tuple, tuple[float, float]] = OrderedDict()


def _fitness_cache_get(key: tuple) -> tuple[float, float] | None:
    val = _fitness_cache.get(key)
    if val is not None:
        _fitness_cache.move_to_end(key)
    return val


def _fitness_cache_put(key: tuple, val: tuple[float, float]) -> None:
    if key in _fitness_cache:
        _fitness_cache.move_to_end(key)
    else:
        if len(_fitness_cache) >= _FITNESS_CACHE_MAX:
            # Evict oldest quarter in one shot
            evict = _FITNESS_CACHE_MAX // 4
            for _ in range(evict):
                _fitness_cache.popitem(last=False)
        _fitness_cache[key] = val


# ---- perf #3: tree-key caching on the individual itself --------------------

def _get_tree_key(ind: gp.PrimitiveTree) -> tuple:
    """Return a cached structural fingerprint; compute and attach if missing."""
    key = getattr(ind, "_tree_key_cache", None)
    if key is None:
        out: list = []
        for node in ind:
            if isinstance(node, gp.Terminal):
                out.append(("T", node.name, node.value))
            else:
                out.append(("P", node.name))
        key = tuple(out)
        try:
            ind._tree_key_cache = key  # type: ignore[attr-defined]
        except AttributeError:
            pass  # slots-only class — cache miss every time, still correct
    return key


def _invalidate_tree_key(ind: gp.PrimitiveTree) -> None:
    """Call whenever the tree structure changes (crossover, mutation)."""
    try:
        del ind._tree_key_cache  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _shallow_clone(ind: gp.PrimitiveTree) -> gp.PrimitiveTree:
    """Shallow-clone an individual; copies fitness and key cache."""
    new = ind.__class__(list(ind))
    if ind.fitness.valid:  # type: ignore[attr-defined]
        new.fitness.values = ind.fitness.values  # type: ignore[attr-defined]
    # Propagate cached key — tree content is identical at this point
    cached_key = getattr(ind, "_tree_key_cache", None)
    if cached_key is not None:
        try:
            new._tree_key_cache = cached_key  # type: ignore[attr-defined]
        except AttributeError:
            pass
    return new


# ---- perf #6: module-level selectors, no per-call env var read -------------

def _best_by_fitness(population: list[gp.PrimitiveTree]) -> gp.PrimitiveTree | None:
    if not population:
        return None
    valid = [
        ind for ind in population
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
    # perf #6: running-best scan instead of building a full valid list
    if _FORCE_SIMPLICITY:
        # Need range stats — one pass to collect, one to score
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

    # Fast path: single running-best scan, no list allocation
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


# ---- perf #7: worker-process initializer builds toolbox once per lifetime --

_worker_toolbox: base.Toolbox | None = None


def _worker_init(
    max_tree_height: int,
    tournament_size: int,
    pset: gp.PrimitiveSet,
) -> None:
    """Run once when a worker process starts; builds and caches the toolbox."""
    global _worker_toolbox
    _worker_toolbox = build_toolbox(
        pset, max_tree_height=max_tree_height, tournament_size=tournament_size
    )


def _evolve_island_worker(args: tuple) -> tuple[list[gp.PrimitiveTree], dict]:
    """Module-level worker for ProcessPoolExecutor.

    Returns the evolved island *and* the local fitness cache accumulated
    during this generation so the main process can merge it back.
    """
    (
        island,
        island_size,
        features_scaled,
        targets_scaled,
        pset,
        max_tree_height,
        parsimony_coefficient,
        crossover_rate,
        mutation_rate,
        const_opt_top_k_ratio,
        rng_seed,
        const_opt_timeout,
        # perf #1: no fitness_cache arg — workers maintain their own local cache
    ) = args

    # perf #7: reuse toolbox built by the initializer; fall back to building if
    # the worker was somehow not initialised (e.g. in tests without initializer).
    toolbox = _worker_toolbox
    if toolbox is None:
        toolbox = build_toolbox(
            pset, max_tree_height=max_tree_height, tournament_size=max_tree_height
        )

    rng = np.random.default_rng(rng_seed)

    # perf #1: worker-local cache — no pickling overhead from the main process
    local_cache: dict[tuple, tuple[float, float]] = {}

    def _eval_cached(ind: gp.PrimitiveTree) -> tuple[float, float]:
        key = _get_tree_key(ind)
        hit = local_cache.get(key)
        if hit is not None:
            return hit
        fit = evaluate_individual(
            ind, pset, features_scaled, targets_scaled, parsimony_coefficient, max_tree_height
        )
        local_cache[key] = fit
        return fit

    offspring = toolbox.select(island, len(island))
    offspring = [_shallow_clone(ind) for ind in offspring]

    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if rng.random() < crossover_rate:
            toolbox.mate(c1, c2)
            for child in (c1, c2):
                if child.fitness.valid:
                    del child.fitness.values
                _invalidate_tree_key(child)  # perf #3

    for mutant in offspring:
        if rng.random() < mutation_rate:
            toolbox.mutate(mutant)
            if mutant.fitness.valid:
                del mutant.fitness.values
            _invalidate_tree_key(mutant)  # perf #3

    for child in offspring:
        if not child.fitness.valid:
            child.fitness.values = _eval_cached(child)

    survivors = tools.selNSGA2(island + offspring, island_size)

    top_k = max(1, int(island_size * const_opt_top_k_ratio))
    elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]
    for ind in elite:
        optimize_constants(ind, pset, features_scaled, targets_scaled, timeout=const_opt_timeout)
        _invalidate_tree_key(ind)  # constants changed → key is stale
        ind.fitness.values = _eval_cached(ind)

    return survivors, local_cache


@dataclass(slots=True)
class SymbolicRegressor(VFPModel):
    """Hybrid symbolic regressor: GP + NSGA-II + island migration + SymPy simplification."""

    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.3
    crossover_rate: float = 0.9
    tournament_size: int = 10
    max_tree_height: int = 6
    parsimony_coefficient: float = 1e-4
    tolerance: float = 1e-4
    seed: int | None = None
    n_islands: int = 4
    migration_interval: int = 5
    migration_size: int = 5
    simplify_interval: int = 10
    basic_arithmetic_only: bool = False
    # perf #2: reduced from 0.50 — run const-opt on top 10% every generation
    const_opt_top_k_ratio: float = 0.10
    # perf #2: run full const-opt pass only every N generations (1 = every gen)
    const_opt_interval: int = 3
    parallel_islands: bool = True
    scale: bool = False
    max_eval_time_seconds: float = 1800.0
    pareto_front_: list[gp.PrimitiveTree] = field(default_factory=list)
    best_individual_: gp.PrimitiveTree | None = None
    _toolbox: base.Toolbox | None = None
    _pset: gp.PrimitiveSet | None = None
    _feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    _target_scaler: StandardScaler = field(default_factory=StandardScaler)
    _executor: ProcessPoolExecutor | None = field(default=None, repr=False)
    _last_features_id: tuple | None = field(default=None, repr=False)
    _built_features_name: tuple[str, ...] | None = field(default=None, repr=False)

    def __str__(self) -> str:
        return "symbolic_regressor"

    def close(self) -> None:
        """Shut down the persistent worker pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def get_fit_details(self) -> dict[str, Any]:
        if self.best_individual_ is None:
            raise ValueError("Model has not been fit yet.")
        return {
            "pareto_size": len(self.pareto_front_),
            "best_fitness": self.best_individual_.fitness.values[0],  # type: ignore[attr-defined]
            "best_complexity": self.best_individual_.fitness.values[1],  # type: ignore[attr-defined]
            "expression": str(self.best_individual_),
        }

    # ---- scaling ------------------------------------------------------------
    def _scale_features(self, features: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if not self.scale:
            return features
        return self._feature_scaler.fit_transform(features) if fit else self._feature_scaler.transform(features)

    def _scale_targets(self, targets: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if not self.scale:
            return np.asarray(targets).flatten()
        arr = np.asarray(targets).flatten().reshape(-1, 1)
        return (self._target_scaler.fit_transform(arr) if fit else self._target_scaler.transform(arr)).ravel()

    def _unscale_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if not self.scale:
            return predictions
        return self._target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()

    # ---- internal helpers ---------------------------------------------------
    def _evaluate_with_cache(
        self,
        ind: gp.PrimitiveTree,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, float]:
        # perf #3: use cached key on individual
        key = _get_tree_key(ind)
        cached = _fitness_cache_get(key)
        if cached is not None:
            return cached
        fit = evaluate_individual(
            ind, self._pset, features, targets,
            self.parsimony_coefficient, self.max_tree_height,
        )
        _fitness_cache_put(key, fit)
        return fit

    def _evolve_one_island(
        self,
        island: list[gp.PrimitiveTree],
        island_size: int,
        features_scaled: np.ndarray,
        targets_scaled: np.ndarray,
        rng: np.random.Generator,
        *,
        run_const_opt: bool,
    ) -> list[gp.PrimitiveTree]:
        """Run one generation on a single island and return the survivors."""
        toolbox = self._toolbox
        assert toolbox is not None

        offspring = toolbox.select(island, len(island))  # type: ignore[attr-defined]
        offspring = [_shallow_clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < self.crossover_rate:
                toolbox.mate(c1, c2)  # type: ignore[attr-defined]
                for child in (c1, c2):
                    if child.fitness.valid:  # type: ignore[attr-defined]
                        del child.fitness.values  # type: ignore[attr-defined]
                    _invalidate_tree_key(child)  # perf #3

        for mutant in offspring:
            if rng.random() < self.mutation_rate:
                toolbox.mutate(mutant)  # type: ignore[attr-defined]
                if mutant.fitness.valid:  # type: ignore[attr-defined]
                    del mutant.fitness.values  # type: ignore[attr-defined]
                _invalidate_tree_key(mutant)  # perf #3

        for child in offspring:
            if not child.fitness.valid:  # type: ignore[attr-defined]
                child.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    child, features_scaled, targets_scaled
                )

        survivors = tools.selNSGA2(island + offspring, island_size)

        # perf #2: const-opt only when scheduled
        if run_const_opt:
            top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
            elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
            for ind in elite:
                optimize_constants(
                    ind, self._pset, features_scaled, targets_scaled,
                    timeout=_CONST_OPT_TIMEOUT,
                )
                _invalidate_tree_key(ind)  # constants changed
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind, features_scaled, targets_scaled
                )

        return survivors

    # ---- main fit loop ------------------------------------------------------
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> SymbolicRegressor:
        rng = np.random.default_rng(self.seed)
        n_features = features.shape[1]

        new_feature_names = (
            tuple(features_name) if features_name
            else tuple(f"ARG{i}" for i in range(n_features))
        )

        # Optimisation 1: skip re-scaling when data is unchanged
        features_id = (features.shape, features.dtype, features.data.tobytes()[:256])
        if self._last_features_id != features_id:
            features_scaled = np.ascontiguousarray(
                self._scale_features(features, fit=True), dtype=np.float64
            )
            targets_scaled = np.ascontiguousarray(
                self._scale_targets(targets, fit=True), dtype=np.float64
            )
            self._last_features_id = features_id
        else:
            features_scaled = np.ascontiguousarray(
                self._scale_features(features, fit=False), dtype=np.float64
            )
            targets_scaled = np.ascontiguousarray(
                self._scale_targets(targets, fit=False), dtype=np.float64
            )

        eval_features_scaled = eval_targets_scaled = None
        if eval_set is not None:
            eval_features_scaled = np.ascontiguousarray(
                self._scale_features(eval_set[0], fit=False), dtype=np.float64
            )
            eval_targets_scaled = np.ascontiguousarray(
                self._scale_targets(eval_set[1], fit=False), dtype=np.float64
            )

        # Optimisation 2: reuse _pset / _toolbox when config is unchanged
        if (
            self._pset is None
            or self._toolbox is None
            or self._built_features_name != new_feature_names
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
                k: getattr(self, k)
                for k in self.__slots__
                if not k.startswith("_") and not k.endswith("_")
            },
        )

        # ---- initial population --------------------------------------------
        logger.debug("Building initial population...")
        seed_individuals = build_seed_individuals(self._pset, n_features)
        islands: list[list[gp.PrimitiveTree]] = []
        for island_idx in range(self.n_islands):
            island = self._toolbox.population(n=island_size)  # type: ignore[attr-defined]
            if island_idx == 0 and seed_individuals:
                n_inject = min(len(seed_individuals), island_size // 2)
                island[:n_inject] = [_shallow_clone(s) for s in seed_individuals[:n_inject]]

            for ind in island:
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind, features_scaled, targets_scaled
                )
            # perf #2: const_opt_top_k_ratio already lower; always run on init
            top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
            elite = sorted(island, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
            for ind in elite:
                optimize_constants(
                    ind, self._pset, features_scaled, targets_scaled,
                    timeout=_CONST_OPT_TIMEOUT,
                )
                _invalidate_tree_key(ind)
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind, features_scaled, targets_scaled
                )
            island = tools.selNSGA2(island, len(island))
            islands.append(island)

        # ---- main loop -----------------------------------------------------
        best_eval_fitness = float("inf")
        patience = max(self.generations // 5, 10)
        patience_counter = 0
        best_islands_snapshot: list[list[gp.PrimitiveTree]] | None = None

        island_rng_seeds = [int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)]

        # perf #7: pass initializer so each worker builds the toolbox once
        if self.parallel_islands and self.n_islands > 1:
            if self._executor is None or self._executor._broken:  # type: ignore[attr-defined]
                self._executor = ProcessPoolExecutor(
                    max_workers=self.n_islands,
                    initializer=_worker_init,
                    initargs=(self.max_tree_height, self.tournament_size, self._pset),
                )
        executor = self._executor if (self.parallel_islands and self.n_islands > 1) else None

        fit_start = time.monotonic()
        fit_deadline = fit_start + self.max_eval_time_seconds
        _gen_timeout_floor = 10.0

        for generation in range(1, self.generations + 1):
            # ---- wall-clock fuse -------------------------------------------
            now = time.monotonic()
            if now >= fit_deadline:
                logger.warning(
                    "fit() wall-clock limit reached (%.0fs); stopping after generation %d",
                    self.max_eval_time_seconds,
                    generation - 1,
                )
                break

            remaining_gens = self.generations - generation + 1
            gen_timeout = max(_gen_timeout_floor, (fit_deadline - now) / remaining_gens)

            # perf #2: throttle const-opt to every const_opt_interval generations
            run_const_opt = (generation % self.const_opt_interval == 0)

            if executor is not None:
                worker_args = [
                    (
                        islands[i],
                        island_size,
                        features_scaled,
                        targets_scaled,
                        self._pset,
                        self.max_tree_height,
                        self.parsimony_coefficient,
                        self.crossover_rate,
                        self.mutation_rate,
                        self.const_opt_top_k_ratio if run_const_opt else 0.0,
                        island_rng_seeds[i],
                        _CONST_OPT_TIMEOUT,
                        # perf #1: no fitness_cache argument
                    )
                    for i in range(self.n_islands)
                ]
                try:
                    futures = [executor.submit(_evolve_island_worker, args) for args in worker_args]
                    results = [f.result(timeout=gen_timeout) for f in futures]
                    islands = [r[0] for r in results]
                    # perf #1: merge worker-local caches back into the main LRU cache
                    for _, worker_cache in results:
                        for k, v in worker_cache.items():
                            _fitness_cache_put(k, v)
                except FuturesTimeoutError:
                    logger.warning(
                        "Generation %d timed out (%.1fs per-gen budget); stopping.",
                        generation, gen_timeout,
                    )
                    for f in futures:
                        f.cancel()
                    break
                island_rng_seeds = [int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)]
            else:
                gen_start = time.monotonic()
                evolved: list[list[gp.PrimitiveTree]] = []
                timed_out = False
                for i in range(self.n_islands):
                    if time.monotonic() - gen_start > gen_timeout:
                        logger.warning(
                            "Serial generation %d timed out (%.1fs); keeping remaining islands.",
                            generation, gen_timeout,
                        )
                        evolved.extend(islands[i:])
                        timed_out = True
                        break
                    evolved.append(
                        self._evolve_one_island(
                            islands[i], island_size, features_scaled, targets_scaled, rng,
                            run_const_opt=run_const_opt,
                        )
                    )
                islands = evolved
                if timed_out:
                    break

            if self.simplify_interval > 0 and generation % self.simplify_interval == 0:
                for island in islands:
                    front = tools.sortNondominated(island, len(island), first_front_only=True)[0]
                    simplify_island(
                        front, self._pset, features_scaled, targets_scaled,
                        n_features, self.parsimony_coefficient, self.max_tree_height,
                    )

            if self.n_islands > 1 and self.migration_interval > 0 and generation % self.migration_interval == 0:
                migrate(islands, self.migration_size, rng)

            best = _best_by_fitness_across_islands(islands)

            if best and best.fitness.values[0] < self.tolerance:  # type: ignore[attr-defined]
                logger.debug(
                    "Early stopping reached (train tolerance)",
                    extra={"generation": generation, "fitness": best.fitness.values[0]},  # type: ignore[attr-defined]
                )
                best_islands_snapshot = islands
                break

            if best is not None and eval_features_scaled is not None and eval_targets_scaled is not None:
                val_fitness, _ = evaluate_individual(
                    best, self._pset, eval_features_scaled, eval_targets_scaled,
                    self.parsimony_coefficient, self.max_tree_height,
                )
                if val_fitness < best_eval_fitness:
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
                            },
                        )
                        break
            # perf #8: only snapshot when there's no eval_set AND we actually
            # need a reference to the current state (i.e. no snapshot yet).
            # The finalization block below falls back gracefully when None.

            logger.debug(
                "Generation complete",
                extra={
                    "generation": generation,
                    "best_fitness": best.fitness.values[0] if best else None,  # type: ignore[attr-defined]
                    "best_complexity": best.fitness.values[1] if best else None,  # type: ignore[attr-defined]
                },
            )

        # ---- finalize ------------------------------------------------------
        # perf #8: best_islands_snapshot may be None when no eval_set was
        # provided and no early-stop fired — just use the final islands.
        if best_islands_snapshot is not None and best_islands_snapshot is not islands:
            islands = best_islands_snapshot
        else:
            islands = [[_shallow_clone(ind) for ind in isl] for isl in islands]

        all_individuals = [ind for island in islands for ind in island]

        if self.simplify_interval > 0:
            front = tools.sortNondominated(all_individuals, len(all_individuals), first_front_only=True)[0]
            simplify_island(
                front, self._pset, features_scaled, targets_scaled,
                n_features, self.parsimony_coefficient, self.max_tree_height,
            )

        self.pareto_front_ = tools.sortNondominated(
            all_individuals, len(all_individuals), first_front_only=True
        )[0]

        self.best_individual_ = _best_by_fitness(self.pareto_front_)
        if self.best_individual_ is None:
            raise RuntimeError("Symbolic regression did not produce a valid model.")

        logger.debug("Symbolic regression complete", extra=self.get_fit_details())
        return self

    # ---- predict ------------------------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.best_individual_ is None or self._pset is None:
            raise ValueError("Model has not been fit yet.")
        features_scaled = self._scale_features(features, fit=False)
        func = gp.compile(self.best_individual_, self._pset)
        predictions_scaled = vectorised_evaluate(func, features_scaled)
        predictions = self._unscale_predictions(predictions_scaled)
        logger.debug("Symbolic prediction complete", extra={"samples": int(features.shape[0])})
        return predictions