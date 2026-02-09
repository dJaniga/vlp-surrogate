from __future__ import annotations

from copy import deepcopy
import logging
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
from deap import base, gp, tools

from vfp.modeling.base import VFPModel
from .primitives import build_primitive_set
from .toolbox import build_toolbox
from .algebraic_simplification import simplify_island
from .helpers import (
    build_seed_individuals,
    optimize_constants,
    evaluate_individual,
    migrate,
    vectorised_evaluate,
)

logger = logging.getLogger(__name__)

# TODO: paramterize
_PENALTY_FITNESS = (1e18, 1e18)


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
    simplify_interval: int = 5
    parsimony_coefficient: float = 0.001
    pareto_front_: list[gp.PrimitiveTree] = field(default_factory=list)
    best_individual_: gp.PrimitiveTree | None = None
    _toolbox: base.Toolbox | None = None
    _pset: gp.PrimitiveSet | None = None
    _feature_mean: np.ndarray | None = None
    _feature_std: np.ndarray | None = None
    _target_mean: float = 0.0
    _target_std: float = 1.0

    def _standardize_features(
        self, features: np.ndarray, *, fit: bool = False
    ) -> np.ndarray:
        """Standardize features to zero-mean, unit-variance.

        When ``fit=True``, computes and stores the scaler parameters from the
        data.  Features with zero variance are left unscaled (std clamped to 1).
        """
        if fit:
            self._feature_mean = features.mean(axis=0)
            self._feature_std = features.std(axis=0)
            # Clamp zero-variance features to avoid division by zero
            self._feature_std[self._feature_std < 1e-12] = 1.0
        assert self._feature_mean is not None and self._feature_std is not None
        return (features - self._feature_mean) / self._feature_std

    def _standardize_targets(self, targets: np.ndarray) -> np.ndarray:
        """Standardize targets to zero-mean, unit-variance and store params."""
        self._target_mean = float(targets.mean())
        self._target_std = float(targets.std())
        if self._target_std < 1e-12:
            self._target_std = 1.0
        return (targets - self._target_mean) / self._target_std

    def _unstandardize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert standardized predictions back to the original target scale."""
        return predictions * self._target_std + self._target_mean

    def fit(self, features: np.ndarray, targets: np.ndarray) -> SymbolicRegressor:
        rng = np.random.default_rng(self.seed)
        n_features = features.shape[1]

        features_std = self._standardize_features(features, fit=True)
        targets_std = self._standardize_targets(targets)

        self._pset = build_primitive_set(n_features)
        self._toolbox = build_toolbox(
            self._pset,
            rng=rng,
            max_tree_height=self.max_tree_height,
            tournament_size=self.tournament_size,
        )

        island_size = self.population_size // self.n_islands
        if island_size < 4:
            raise ValueError(
                f"population_size={self.population_size} is too small for "
                f"n_islands={self.n_islands} (need at least 4 per island)."
            )

        logger.info(
            "Initializing symbolic regression",
            extra={
                "population": self.population_size,
                "generations": self.generations,
                "islands": self.n_islands,
                "island_size": island_size,
            },
        )

        seed_individuals = build_seed_individuals(self._pset, n_features)
        islands: list[list[gp.PrimitiveTree]] = []
        for island_idx in range(self.n_islands):
            island = self._toolbox.population(n=island_size)  # type: ignore[attr-defined]
            # Inject seed individuals into the first island(s)
            if island_idx == 0 and seed_individuals:
                n_inject = min(len(seed_individuals), island_size // 2)
                island[:n_inject] = [deepcopy(s) for s in seed_individuals[:n_inject]]
            for ind in island:
                optimize_constants(ind, self._pset, features_std, targets_std)
                ind.fitness.values = evaluate_individual(  # type: ignore[attr-defined]
                    ind,
                    self._pset,
                    features_std,
                    targets_std,
                    self.parsimony_coefficient,
                )
            island = tools.selNSGA2(island, len(island))
            islands.append(island)

        for generation in tqdm(range(1, self.generations + 1)):
            for island_idx, island in enumerate(islands):
                offspring = self._toolbox.select(island, len(island))  # type: ignore[attr-defined]
                offspring = [deepcopy(ind) for ind in offspring]

                # crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if rng.random() < self.crossover_rate:
                        self._toolbox.mate(child1, child2)  # type: ignore[attr-defined]
                        del child1.fitness.values, child2.fitness.values  # type: ignore[attr-defined]

                # mutation
                for mutant in offspring:
                    if rng.random() < self.mutation_rate:
                        self._toolbox.mutate(mutant)  # type: ignore[attr-defined]
                        del mutant.fitness.values  # type: ignore[attr-defined]

                # evaluate invalidated
                for child in offspring:
                    if not child.fitness.valid:  # type: ignore[attr-defined]
                        optimize_constants(child, self._pset, features_std, targets_std)
                        child.fitness.values = evaluate_individual(  # type: ignore[attr-defined]
                            child,
                            self._pset,
                            features_std,
                            targets_std,
                            self.parsimony_coefficient,
                        )

                islands[island_idx] = tools.selNSGA2(
                    island + offspring,
                    island_size,
                )

            # periodic simplification
            if self.simplify_interval > 0 and generation % self.simplify_interval == 0:
                for island in islands:
                    simplify_island(
                        island,
                        self._pset,
                        features_std,
                        targets_std,
                        self.parsimony_coefficient,
                        n_features,
                    )

            # periodic migration
            if (
                self.n_islands > 1
                and self.migration_interval > 0
                and generation % self.migration_interval == 0
            ):
                migrate(islands, self.migration_size, rng)

            all_individuals = [ind for island in islands for ind in island]
            best = _best_by_mse(all_individuals)
            if best and best.fitness.values[0] < self.tolerance:  # type: ignore[attr-defined]
                logger.info(
                    "Early stopping reached",
                    extra={
                        "generation": generation,
                        "mse": best.fitness.values[0],  # type: ignore[attr-defined]
                    },
                )
                break

            logger.debug(
                "Generation complete",
                extra={
                    "generation": generation,
                    "best_mse": best.fitness.values[0] if best else None,  # type: ignore[attr-defined]
                    "best_len": len(best) if best else None,
                },
            )

        all_individuals = [ind for island in islands for ind in island]

        # simplification pass on the full population
        if self.simplify_interval > 0:
            simplify_island(
                all_individuals,
                self._pset,
                features_std,
                targets_std,
                self.parsimony_coefficient,
                n_features,
            )

        self.pareto_front_ = tools.sortNondominated(
            all_individuals,
            len(all_individuals),
            first_front_only=True,
        )[0]
        logger.debug(
            "Pareto front extracted",
            extra={"size": len(self.pareto_front_)},
        )
        self.best_individual_ = _best_by_mse(self.pareto_front_)
        if self.best_individual_ is None:
            raise RuntimeError("Symbolic regression did not produce a valid model.")

        logger.info(
            "Symbolic regression complete",
            extra={
                "pareto_size": len(self.pareto_front_),
                "best_mse": self.best_individual_.fitness.values[0],  # type: ignore[attr-defined]
                "best_complexity": self.best_individual_.fitness.values[1],  # type: ignore[attr-defined]
                "expression": str(self.best_individual_),
            },
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.best_individual_ is None or self._pset is None:
            raise ValueError("Model has not been fit yet.")
        features_std = self._standardize_features(features)
        func = gp.compile(self.best_individual_, self._pset)
        predictions_std = vectorised_evaluate(func, features_std)
        predictions = self._unstandardize_predictions(predictions_std)
        logger.info(
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
