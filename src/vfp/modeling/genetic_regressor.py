from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from deap import algorithms, base, creator, tools

from vfp.modeling.base import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneticAlgorithmRegressor(VFPModel):
    population_size: int = 60
    generations: int = 80
    crossover_prob: float = 0.7
    mutation_prob: float = 0.25
    tournament_size: int = 3
    seed: int | None = None
    coeff_bounds: tuple[float, float] = (-1.0, 1.0)
    coefficients: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray
    ) -> GeneticAlgorithmRegressor:
        logger.info(
            "Starting GA training",
            extra={
                "population": self.population_size,
                "generations": self.generations,
                "features": int(features.shape[1]),
            },
        )
        rng = np.random.default_rng(self.seed)
        design_matrix = np.column_stack([np.ones(features.shape[0]), features])
        coeff_count = design_matrix.shape[1]

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore[attr-defined]

        logger.debug(
            "GA toolbox configured",
            extra={"coeff_count": coeff_count, "bounds": self.coeff_bounds},
        )

        toolbox = base.Toolbox()
        toolbox.register(
            "coeff", rng.uniform, self.coeff_bounds[0], self.coeff_bounds[1]
        )
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,  # type: ignore[attr-defined]
            toolbox.coeff,  # type: ignore[attr-defined]
            n=coeff_count,
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,  # type: ignore[attr-defined]
        )
        toolbox.register("evaluate", _mse_fitness, design_matrix, targets)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        population = toolbox.population(n=self.population_size)  # type: ignore[attr-defined]
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            verbose=False,
        )

        best = tools.selBest(population, k=1)[0]
        self.coefficients = np.asarray(best, dtype=float)
        logger.info(
            "GA training complete",
            extra={
                "coefficients": self.coefficients.tolist(),
                "coeff_count": coeff_count,
            },
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model has not been fit yet.")
        logger.info(
            "Predicting with GA model",
            extra={"samples": int(features.shape[0])},
        )
        design_matrix = np.column_stack([np.ones(features.shape[0]), features])
        return design_matrix @ self.coefficients


def _mse_fitness(
    design_matrix: np.ndarray, targets: np.ndarray, coefficients: list[float]
) -> tuple[float]:
    predictions = design_matrix @ np.asarray(coefficients)
    mse = float(np.mean((predictions - targets) ** 2))
    return (mse,)
