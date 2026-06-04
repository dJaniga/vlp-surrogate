from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from deap import base, gp, tools
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from vfp.modeling.base import VFPModel
from .algebraic_simplification import simplify_island
from .helpers import (
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


def _shallow_clone(ind: gp.PrimitiveTree) -> gp.PrimitiveTree:
    """Shallow-clone an individual: the node list is copied, nodes themselves
    are immutable (Primitive/Terminal). Fitness is also copied so that we keep
    a valid value across selection without the cost of a full deepcopy.
    """
    new = ind.__class__(list(ind))
    if ind.fitness.valid:  # type: ignore[attr-defined]
        new.fitness.values = ind.fitness.values  # type: ignore[attr-defined]
    return new


def _tree_key(ind: gp.PrimitiveTree) -> tuple:
    """A hashable structural fingerprint of an individual.

    Two individuals with the same fingerprint produce identical predictions
    on any input (constants and primitives are fully captured).
    """
    out: list = []
    for node in ind:
        if isinstance(node, gp.Terminal):
            out.append(("T", node.name, node.value))
        else:
            out.append(("P", node.name))
    return tuple(out)


@dataclass(slots=True)
class SymbolicRegressor(VFPModel):
    """Hybrid symbolic regressor: GP + NSGA-II + island migration + SymPy simplification."""

    population_size: int = 200
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 3
    max_tree_height: int = 6
    tolerance: float = 1e-6
    seed: int | None = None
    n_islands: int = 5
    migration_interval: int = 10
    migration_size: int = 3
    simplify_interval: int = 15  # less frequent: SymPy is expensive
    basic_arithmetic_only: bool = False
    const_opt_top_k_ratio: float = 0.25  # only optimize constants for top quartile
    parallel_islands: bool = True
    # Scaling: set to False when an external scaler (e.g. ModelWrapper) handles it
    scale: bool = False
    pareto_front_: list[gp.PrimitiveTree] = field(default_factory=list)
    best_individual_: gp.PrimitiveTree | None = None
    _toolbox: base.Toolbox | None = None
    _pset: gp.PrimitiveSet | None = None
    _feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    _target_scaler: StandardScaler = field(default_factory=StandardScaler)
    # Fitness cache: structural fingerprint -> (mse_penalised, complexity)
    # Persisted across fit() calls to warm-start subsequent runs.
    _fitness_cache: dict = field(default_factory=dict)
    # Reused across fit() calls to avoid process-spawn overhead in hot loops.
    _executor: ProcessPoolExecutor | None = field(default=None, repr=False)
    # Fingerprint of the last training data seen; used to skip re-scaling.
    _last_features_id: tuple | None = field(default=None, repr=False)
    # Feature names used when the current _pset / _toolbox were built.
    _built_features_name: tuple[str, ...] | None = field(default=None, repr=False)

    def __str__(self) -> str:
        return "symbolic_regressor"

    def close(self) -> None:
        """Shut down the persistent worker pool. Call when done with hot-loop fitting."""
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
        if fit:
            return self._feature_scaler.fit_transform(features)
        return self._feature_scaler.transform(features)

    def _scale_targets(self, targets: np.ndarray, *, fit: bool = False) -> np.ndarray:
        if not self.scale:
            return np.asarray(targets).flatten()
        targets = np.asarray(targets).flatten()
        arr = targets.reshape(-1, 1)
        if fit:
            return self._target_scaler.fit_transform(arr).ravel()
        return self._target_scaler.transform(arr).ravel()

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
        key = _tree_key(ind)
        cached = self._fitness_cache.get(key)
        if cached is not None:
            return cached
        fit = evaluate_individual(ind, self._pset, features, targets)
        if len(self._fitness_cache) < 50_000:
            self._fitness_cache[key] = fit
        return fit

    def _evolve_one_island(
        self,
        island: list[gp.PrimitiveTree],
        island_size: int,
        features_scaled: np.ndarray,
        targets_scaled: np.ndarray,
        rng: np.random.Generator,
    ) -> list[gp.PrimitiveTree]:
        """Run one generation on a single island and return the survivors."""
        toolbox = self._toolbox
        assert toolbox is not None

        offspring = toolbox.select(island, len(island))  # type: ignore[attr-defined]
        offspring = [_shallow_clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < self.crossover_rate:
                toolbox.mate(c1, c2)  # type: ignore[attr-defined]
                if c1.fitness.valid:  # type: ignore[attr-defined]
                    del c1.fitness.values  # type: ignore[attr-defined]
                if c2.fitness.valid:  # type: ignore[attr-defined]
                    del c2.fitness.values  # type: ignore[attr-defined]

        for mutant in offspring:
            if rng.random() < self.mutation_rate:
                toolbox.mutate(mutant)  # type: ignore[attr-defined]
                if mutant.fitness.valid:  # type: ignore[attr-defined]
                    del mutant.fitness.values  # type: ignore[attr-defined]

        for child in offspring:
            if not child.fitness.valid:  # type: ignore[attr-defined]
                child.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    child, features_scaled, targets_scaled
                )

        survivors = tools.selNSGA2(island + offspring, island_size)

        top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
        elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
        for ind in elite:
            optimize_constants(ind, self._pset, features_scaled, targets_scaled)
            new_fit = self._evaluate_with_cache(ind, features_scaled, targets_scaled)
            ind.fitness.values = new_fit  # type: ignore[attr-defined]

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
            tuple(features_name)
            if features_name
            else tuple(f"ARG{i}" for i in range(n_features))
        )

        # ---- Optimization 1: skip re-scaling when data is unchanged ----------
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

        eval_features_scaled = None
        eval_targets_scaled = None
        if eval_set is not None:
            eval_features_scaled = np.ascontiguousarray(
                self._scale_features(eval_set[0], fit=False), dtype=np.float64
            )
            eval_targets_scaled = np.ascontiguousarray(
                self._scale_targets(eval_set[1], fit=False), dtype=np.float64
            )

        # ---- Optimization 2: reuse _pset / _toolbox when config is unchanged -
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
        seed_individuals = build_seed_individuals(self._pset, n_features)
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
                    ind, features_scaled, targets_scaled
                )
            top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
            elite = sorted(island, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
            for ind in elite:
                optimize_constants(ind, self._pset, features_scaled, targets_scaled)
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

        island_rng_seeds = [
            int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)
        ]

        # ---- Optimization 3: reuse persistent executor across fit() calls ---
        if self.parallel_islands and self.n_islands > 1:
            if self._executor is None or self._executor._broken:  # type: ignore[attr-defined]
                self._executor = ProcessPoolExecutor(max_workers=self.n_islands)
        executor = (
            self._executor if (self.parallel_islands and self.n_islands > 1) else None
        )

        for generation in range(1, self.generations + 1):
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
                        self.crossover_rate,
                        self.mutation_rate,
                        self.const_opt_top_k_ratio,
                        dict(self._fitness_cache),
                        island_rng_seeds[i],
                    )
                    for i in range(self.n_islands)
                ]
                islands = list(executor.map(_evolve_island_worker, worker_args))
                island_rng_seeds = [
                    int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)
                ]
            else:
                islands = [
                    self._evolve_one_island(
                        islands[i],
                        island_size,
                        features_scaled,
                        targets_scaled,
                        rng,
                    )
                    for i in range(self.n_islands)
                ]

            if self.simplify_interval > 0 and generation % self.simplify_interval == 0:
                for island in islands:
                    front = tools.sortNondominated(
                        island, len(island), first_front_only=True
                    )[0]
                    simplify_island(
                        front,
                        self._pset,
                        features_scaled,
                        targets_scaled,
                        n_features,
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
                )

                if val_fitness < best_eval_fitness:
                    best_eval_fitness = val_fitness
                    patience_counter = 0
                    # ---- Optimization 4: shallow-clone snapshot instead of deepcopy --
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
            else:
                best_islands_snapshot = islands

            logger.debug(
                "Generation complete",
                extra={
                    "generation": generation,
                    "best_fitness": best.fitness.values[0] if best else None,  # type: ignore[attr-defined]
                    "best_complexity": best.fitness.values[1] if best else None, # type: ignore[attr-defined]
                },
            )

        # ---- finalize ------------------------------------------------------
        if best_islands_snapshot is not None and best_islands_snapshot is not islands:
            islands = best_islands_snapshot
        elif best_islands_snapshot is islands:
            islands = [[_shallow_clone(ind) for ind in isl] for isl in islands]

        all_individuals = [ind for island in islands for ind in island]

        if self.simplify_interval > 0:
            front = tools.sortNondominated(
                all_individuals, len(all_individuals), first_front_only=True
            )[0]
            simplify_island(
                front,
                self._pset,
                features_scaled,
                targets_scaled,
                n_features,
            )

        self.pareto_front_ = tools.sortNondominated(
            all_individuals, len(all_individuals), first_front_only=True
        )[0]
        self.best_individual_ = _best_by_fitness(self.pareto_front_)
        if self.best_individual_ is None:
            raise RuntimeError("Symbolic regression did not produce a valid model.")

        # ---- Optimization 5: trim cache instead of clearing it --------------
        if len(self._fitness_cache) > 40_000:
            self._fitness_cache.clear()

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
        logger.debug(
            "Symbolic prediction complete",
            extra={"samples": int(features.shape[0])},
        )
        return predictions


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
    return min(valid, key=lambda ind: ind.fitness.values[0])  # type: ignore[attr-defined]

# def _best_by_fitness(population: list[gp.PrimitiveTree]) -> gp.PrimitiveTree | None:
#     if not population:
#         return None
#     valid = [
#         ind for ind in population
#         if ind.fitness.valid and ind.fitness.values[0] < _PENALTY_FITNESS[0]
#     ]
#     if not valid:
#         return min(population, key=lambda ind: ind.fitness.values[0])
#
#     errors = [ind.fitness.values[0] for ind in valid]
#     complexities = [ind.fitness.values[1] for ind in valid]
#
#     min_e, max_e = min(errors), max(errors)
#     min_c, max_c = min(complexities), max(complexities)
#
#     def normalised_score(ind):
#         e = ind.fitness.values[0]
#         c = ind.fitness.values[1]
#         norm_e = (e - min_e) / (max_e - min_e + 1e-12)
#         norm_c = (c - min_c) / (max_c - min_c + 1e-12)
#         return norm_e + norm_c  # equal weight; tune as needed
#
#     return min(valid, key=normalised_score)


def _best_by_fitness_across_islands(
    islands: list[list[gp.PrimitiveTree]],
) -> gp.PrimitiveTree | None:
    """Avoid building a flat list of all individuals just to find the min."""
    best: gp.PrimitiveTree | None = None
    best_fitness = float("inf")
    for island in islands:
        for ind in island:
            if not ind.fitness.valid:  # type: ignore[attr-defined]
                continue
            fitness = ind.fitness.values[0]  # type: ignore[attr-defined]
            if fitness < best_fitness and fitness < _PENALTY_FITNESS[0]:
                best_fitness = fitness
                best = ind
    return best

# def _best_by_fitness_across_islands(
#     islands: list[list[gp.PrimitiveTree]],
# ) -> gp.PrimitiveTree | None:
#     """Find the best individual across all islands using the same normalised
#     score as _best_by_fitness, so generation logs are consistent with the
#     final result.
#     """
#     all_valid = [
#         ind
#         for island in islands
#         for ind in island
#         if ind.fitness.valid and ind.fitness.values[0] < _PENALTY_FITNESS[0]  # type: ignore[attr-defined]
#     ]
#     if not all_valid:
#         return None
#
#     errors = [ind.fitness.values[0] for ind in all_valid]  # type: ignore[attr-defined]
#     complexities = [ind.fitness.values[1] for ind in all_valid]  # type: ignore[attr-defined]
#
#     min_e, max_e = min(errors), max(errors)
#     min_c, max_c = min(complexities), max(complexities)
#
#     def normalised_score(ind: gp.PrimitiveTree) -> float:
#         e = ind.fitness.values[0]  # type: ignore[attr-defined]
#         c = ind.fitness.values[1]  # type: ignore[attr-defined]
#         norm_e = (e - min_e) / (max_e - min_e + 1e-12)
#         norm_c = (c - min_c) / (max_c - min_c + 1e-12)
#         return norm_e + norm_c
#
#     return min(all_valid, key=normalised_score)


def _evolve_island_worker(
    args: tuple,
) -> list[gp.PrimitiveTree]:
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    (
        island,
        island_size,
        features_scaled,
        targets_scaled,
        pset,
        max_tree_height,
        tournament_size,
        crossover_rate,
        mutation_rate,
        const_opt_top_k_ratio,
        fitness_cache,
        rng_seed,
    ) = args

    toolbox = build_toolbox(
        pset, max_tree_height=max_tree_height, tournament_size=tournament_size
    )
    rng = np.random.default_rng(rng_seed)

    def _eval_cached(ind):
        key = _tree_key(ind)
        if key in fitness_cache:
            return fitness_cache[key]
        fit = evaluate_individual(ind, pset, features_scaled, targets_scaled)
        fitness_cache[key] = fit
        return fit

    offspring = toolbox.select(island, len(island))
    offspring = [_shallow_clone(ind) for ind in offspring]

    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if rng.random() < crossover_rate:
            toolbox.mate(c1, c2)
            if c1.fitness.valid:
                del c1.fitness.values
            if c2.fitness.valid:
                del c2.fitness.values

    for mutant in offspring:
        if rng.random() < mutation_rate:
            toolbox.mutate(mutant)
            if mutant.fitness.valid:
                del mutant.fitness.values

    for child in offspring:
        if not child.fitness.valid:
            child.fitness.values = _eval_cached(child)

    survivors = tools.selNSGA2(island + offspring, island_size)

    top_k = max(1, int(island_size * const_opt_top_k_ratio))
    elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]
    for ind in elite:
        optimize_constants(ind, pset, features_scaled, targets_scaled)
        ind.fitness.values = _eval_cached(ind)

    return survivors
