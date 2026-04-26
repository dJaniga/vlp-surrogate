from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from deap import base, gp, tools
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
            # value differentiates ARGn from constants
            out.append(("T", node.name, node.value))
        else:  # Primitive
            out.append(("P", node.name))
    return tuple(out)


@dataclass(slots=True)
class SymbolicRegressor(VFPModel):
    """Hybrid symbolic regressor: GP + NSGA-II + island migration + SymPy simplification."""

    population_size: int = 120
    generations: int = 30
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 3
    max_tree_height: int = 6
    tolerance: float = 1e-4
    seed: int | None = None
    n_islands: int = 4
    migration_interval: int = 5
    migration_size: int = 3
    simplify_interval: int = 10  # less frequent: SymPy is expensive
    parsimony_coefficient: float = 0.001
    basic_arithmetic_only: bool = False
    # New tunables
    const_opt_top_k_ratio: float = 0.25  # only optimize constants for top quartile
    parallel_islands: bool = True
    pareto_front_: list[gp.PrimitiveTree] = field(default_factory=list)
    best_individual_: gp.PrimitiveTree | None = None
    _toolbox: base.Toolbox | None = None
    _pset: gp.PrimitiveSet | None = None
    _feature_mean: np.ndarray | None = None
    _feature_std: np.ndarray | None = None
    _target_mean: float = 0.0
    _target_std: float = 1.0
    # Fitness cache: structural fingerprint -> (mse_penalised, complexity)
    _fitness_cache: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return "symbolic_regressor"

    def get_fit_details(self) -> dict[str, Any]:
        if self.best_individual_ is None:
            raise ValueError("Model has not been fit yet.")
        return {
            "pareto_size": len(self.pareto_front_),
            "best_mse": self.best_individual_.fitness.values[0],  # type: ignore[attr-defined]
            "best_complexity": self.best_individual_.fitness.values[1],  # type: ignore[attr-defined]
            "expression": str(self.best_individual_),
        }

    # ---- standardization (unchanged behavior) -------------------------------
    def _standardize_features(
        self, features: np.ndarray, *, fit: bool = False
    ) -> np.ndarray:
        if fit:
            self._feature_mean = features.mean(axis=0)
            _std = features.std(axis=0)
            _std[_std < 1e-12] = 1.0
            self._feature_std = _std
        assert self._feature_mean is not None and self._feature_std is not None
        return (features - self._feature_mean) / self._feature_std

    def _standardize_targets(self, targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets).flatten()
        self._target_mean = float(targets.mean())
        self._target_std = float(targets.std())
        if self._target_std < 1e-12:
            self._target_std = 1.0
        return (targets - self._target_mean) / self._target_std

    def _unstandardize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        return predictions * self._target_std + self._target_mean

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
        fit = evaluate_individual(
            ind, self._pset, features, targets, self.parsimony_coefficient
        )
        # bound cache size to avoid memory blow-up on very long runs
        if len(self._fitness_cache) < 50_000:
            self._fitness_cache[key] = fit
        return fit

    def _evolve_one_island(
        self,
        island: list[gp.PrimitiveTree],
        island_size: int,
        features_std: np.ndarray,
        targets_std: np.ndarray,
        rng: np.random.Generator,
    ) -> list[gp.PrimitiveTree]:
        """Run one generation on a single island and return the survivors."""
        toolbox = self._toolbox
        assert toolbox is not None

        # --- selection (shallow-clone is enough; nodes are immutable) -------
        offspring = toolbox.select(island, len(island))  # type: ignore[attr-defined]
        offspring = [_shallow_clone(ind) for ind in offspring]

        # --- crossover ------------------------------------------------------
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < self.crossover_rate:
                toolbox.mate(c1, c2)  # type: ignore[attr-defined]
                if c1.fitness.valid:  # type: ignore[attr-defined]
                    del c1.fitness.values  # type: ignore[attr-defined]
                if c2.fitness.valid:  # type: ignore[attr-defined]
                    del c2.fitness.values  # type: ignore[attr-defined]

        # --- mutation -------------------------------------------------------
        for mutant in offspring:
            if rng.random() < self.mutation_rate:
                toolbox.mutate(mutant)  # type: ignore[attr-defined]
                if mutant.fitness.valid:  # type: ignore[attr-defined]
                    del mutant.fitness.values  # type: ignore[attr-defined]

        # --- cheap fitness only (no constant optimization yet) --------------
        for child in offspring:
            if not child.fitness.valid:  # type: ignore[attr-defined]
                child.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    child, features_std, targets_std
                )

        # --- pick survivors -------------------------------------------------
        survivors = tools.selNSGA2(island + offspring, island_size)

        # --- expensive constant optimization on the elite only --------------
        top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
        # Sort by primary objective (MSE+parsimony); already cheap on small list
        elite = sorted(survivors, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
        for ind in elite:
            optimize_constants(ind, self._pset, features_std, targets_std)
            new_fit = self._evaluate_with_cache(ind, features_std, targets_std)
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

        # Ensure contiguous float64 arrays for fast NumPy ops
        features_std = np.ascontiguousarray(
            self._standardize_features(features, fit=True), dtype=np.float64
        )
        targets_std = np.ascontiguousarray(
            self._standardize_targets(targets), dtype=np.float64
        )

        eval_features_std = None
        eval_targets_std = None
        if eval_set is not None:
            eval_features_std = np.ascontiguousarray(
                self._standardize_features(eval_set[0], fit=False), dtype=np.float64
            )
            _t = np.asarray(eval_set[1]).flatten()
            eval_targets_std = np.ascontiguousarray(
                (_t - self._target_mean) / self._target_std, dtype=np.float64
            )

        # ---- build pset / toolbox -----------------------------------------
        self._pset = build_primitive_set(n_features, self.basic_arithmetic_only)
        self.features_name = tuple(
            features_name
            if features_name
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )
        self._pset.renameArguments(
            **{f"ARG{idx}": name for idx, name in enumerate(self.features_name)}
        )
        self._toolbox = build_toolbox(
            self._pset,
            max_tree_height=self.max_tree_height,
            tournament_size=self.tournament_size,
        )

        island_size = self.population_size // self.n_islands
        if island_size < 4:
            raise ValueError(
                f"population_size={self.population_size} is too small for "
                f"n_islands={self.n_islands} (need at least 4 per island)."
            )

        logger.debug(
            "Initializing symbolic regression",
            extra={
                "population": self.population_size,
                "generations": self.generations,
                "islands": self.n_islands,
                "island_size": island_size,
                "parsimony_coefficient": self.parsimony_coefficient,
            },
        )

        # ---- initial population --------------------------------------------
        seed_individuals = build_seed_individuals(self._pset, n_features)
        islands: list[list[gp.PrimitiveTree]] = []
        for island_idx in range(self.n_islands):
            island = self._toolbox.population(n=island_size)  # type: ignore[attr-defined]
            if island_idx == 0 and seed_individuals:
                n_inject = min(len(seed_individuals), island_size // 2)
                island[:n_inject] = [_shallow_clone(s) for s in seed_individuals[:n_inject]]

            # Evaluate cheaply first
            for ind in island:
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind, features_std, targets_std
                )
            # Optimize constants only on top-K of initial island
            top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
            elite = sorted(island, key=lambda i: i.fitness.values[0])[:top_k]  # type: ignore[attr-defined]
            for ind in elite:
                optimize_constants(ind, self._pset, features_std, targets_std)
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind, features_std, targets_std
                )
            island = tools.selNSGA2(island, len(island))
            islands.append(island)

        # ---- main loop -----------------------------------------------------
        best_eval_mse = float("inf")
        patience = max(self.generations // 10, 10)
        patience_counter = 0
        best_islands_snapshot: list[list[gp.PrimitiveTree]] | None = None

        # Per-island RNGs for thread safety
        island_rngs = [
            np.random.default_rng(rng.integers(0, 2**63 - 1))
            for _ in range(self.n_islands)
        ]

        executor: ThreadPoolExecutor | None = None
        if self.parallel_islands and self.n_islands > 1:
            executor = ThreadPoolExecutor(max_workers=self.n_islands)

        try:
            for generation in tqdm(range(1, self.generations + 1)):
                # ---- evolve all islands (parallel or serial) ---------------
                if executor is not None:
                    futures = [
                        executor.submit(
                            self._evolve_one_island,
                            islands[i],
                            island_size,
                            features_std,
                            targets_std,
                            island_rngs[i],
                        )
                        for i in range(self.n_islands)
                    ]
                    islands = [f.result() for f in futures]
                else:
                    for i in range(self.n_islands):
                        islands[i] = self._evolve_one_island(
                            islands[i],
                            island_size,
                            features_std,
                            targets_std,
                            island_rngs[i],
                        )

                # ---- periodic simplification (Pareto-front only) ----------
                if (
                    self.simplify_interval > 0
                    and generation % self.simplify_interval == 0
                ):
                    for island in islands:
                        # Only simplify the non-dominated front of each island
                        front = tools.sortNondominated(
                            island, len(island), first_front_only=True
                        )[0]
                        simplify_island(
                            front,
                            self._pset,
                            features_std,
                            targets_std,
                            self.parsimony_coefficient,
                            n_features,
                        )

                # ---- periodic migration -----------------------------------
                if (
                    self.n_islands > 1
                    and self.migration_interval > 0
                    and generation % self.migration_interval == 0
                ):
                    migrate(islands, self.migration_size, rng)

                # ---- find current best ------------------------------------
                best = _best_by_mse_across_islands(islands)

                # train tolerance
                if best and best.fitness.values[0] < self.tolerance:  # type: ignore[attr-defined]
                    logger.debug(
                        "Early stopping reached (train tolerance)",
                        extra={
                            "generation": generation,
                            "mse": best.fitness.values[0],  # type: ignore[attr-defined]
                        },
                    )
                    best_islands_snapshot = islands  # lazy: deepcopy only at the end
                    break

                # validation-based early stopping
                if (
                    best is not None
                    and eval_features_std is not None
                    and eval_targets_std is not None
                ):
                    val_mse, _ = evaluate_individual(
                        best,
                        self._pset,
                        eval_features_std,
                        eval_targets_std,
                        self.parsimony_coefficient,
                    )

                    if val_mse < best_eval_mse:
                        best_eval_mse = val_mse
                        patience_counter = 0
                        # Deepcopy ONLY when we actually beat the previous best
                        best_islands_snapshot = [
                            [deepcopy(ind) for ind in island] for island in islands
                        ]
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.debug(
                                "Early stopping reached (validation patience)",
                                extra={
                                    "generation": generation,
                                    "val_mse": val_mse,
                                    "best_val_mse": best_eval_mse,
                                },
                            )
                            break
                else:
                    # No validation set: keep a lazy reference, deepcopy at exit
                    best_islands_snapshot = islands

                logger.debug(
                    "Generation complete",
                    extra={
                        "generation": generation,
                        "best_mse": best.fitness.values[0] if best else None,  # type: ignore[attr-defined]
                        "best_len": len(best) if best else None,
                        "cache_size": len(self._fitness_cache),
                    },
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        # ---- finalize ------------------------------------------------------
        if best_islands_snapshot is not None and best_islands_snapshot is not islands:
            islands = best_islands_snapshot
        elif best_islands_snapshot is islands:
            # Lazy reference path: deepcopy now to detach from any aliases
            islands = [[deepcopy(ind) for ind in isl] for isl in islands]

        all_individuals = [ind for island in islands for ind in island]

        # Final simplification on the full Pareto front only
        if self.simplify_interval > 0:
            front = tools.sortNondominated(
                all_individuals, len(all_individuals), first_front_only=True
            )[0]
            simplify_island(
                front,
                self._pset,
                features_std,
                targets_std,
                self.parsimony_coefficient,
                n_features,
            )

        self.pareto_front_ = tools.sortNondominated(
            all_individuals, len(all_individuals), first_front_only=True
        )[0]
        self.best_individual_ = _best_by_mse(self.pareto_front_)
        if self.best_individual_ is None:
            raise RuntimeError("Symbolic regression did not produce a valid model.")

        # Drop the cache after fit to free memory
        self._fitness_cache.clear()

        logger.debug("Symbolic regression complete", extra=self.get_fit_details())
        return self

    # ---- predict (unchanged) -----------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.best_individual_ is None or self._pset is None:
            raise ValueError("Model has not been fit yet.")
        features_std = self._standardize_features(features)
        func = gp.compile(self.best_individual_, self._pset)
        predictions_std = vectorised_evaluate(func, features_std)
        predictions = self._unstandardize_predictions(predictions_std)
        logger.debug(
            "Symbolic prediction complete",
            extra={"samples": int(features.shape[0])},
        )
        return predictions


def _best_by_mse(population: list[gp.PrimitiveTree]) -> gp.PrimitiveTree | None:
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


def _best_by_mse_across_islands(
    islands: list[list[gp.PrimitiveTree]],
) -> gp.PrimitiveTree | None:
    """Avoid building a flat list of all individuals just to find the min."""
    best: gp.PrimitiveTree | None = None
    best_mse = float("inf")
    for island in islands:
        for ind in island:
            if not ind.fitness.valid:  # type: ignore[attr-defined]
                continue
            mse = ind.fitness.values[0]  # type: ignore[attr-defined]
            if mse < best_mse and mse < _PENALTY_FITNESS[0]:
                best_mse = mse
                best = ind
    return best