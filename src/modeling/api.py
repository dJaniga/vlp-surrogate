from modeling.base import VFPModel
from modeling.genetic_regressor import GeneticAlgorithmRegressor
from modeling.linear_regression import LinearRegressionModel
from modeling.symbolic import SymbolicRegressor
from models import ModelName


def create_model(name: ModelName, **kwargs) -> VFPModel:
    if name == "linear":
        return LinearRegressionModel(**kwargs)
    if name == "ga":
        return GeneticAlgorithmRegressor(**kwargs)
    if name == "symbolic":
        return SymbolicRegressor(**kwargs)
    raise ValueError(f"Unsupported model name: {name}")
