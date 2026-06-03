from __future__ import annotations

import copy
import logging
import numpy as np
import optuna

from sklearn.model_selection import KFold

from vfp.modeling.base import VFPModel
from vfp.modeling.sklearn_regressors.bayesian_ridge_regressor import (
    BayesianRidgeRegressor,
)
from vfp.modeling.sklearn_regressors.elastic_net_regressor import ElasticNetRegressor
from vfp.modeling.sklearn_regressors.xgboost_regressor import XGBoostRegressor
from vfp.modeling.symbolic.symbolic_regressor import SymbolicRegressor
from vfp.modeling.sklearn_regressors.huber_regressor import HuberRegressor
from vfp.modeling.tuning_metrics import evaluate_metric, get_metric_direction

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    model: VFPModel,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: tuple[str, ...],
    n_trials: int = 50,
    n_splits: int = 3,
    tuning_metric: str = "root_mean_squared_error",
    seed: int | None = None,
) -> VFPModel:
    if not isinstance(
            model,
            (
                    XGBoostRegressor,
                    SymbolicRegressor,
                    ElasticNetRegressor,
                    BayesianRidgeRegressor,
                    HuberRegressor,
            ),
    ):
        logger.warning(
            f"Hyperparameter tuning not implemented for {type(model).__name__}. Returning original model."
        )
        return model

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = get_metric_direction(tuning_metric)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        cv_scores = []

        logger.debug(f"Running trial {trial.number + 1} of {n_trials}")

        for idx, (train_idx, val_idx) in enumerate(kf.split(features)):
            logger.debug(
                f"Running inner CV fold {idx + 1} of {n_splits} for {type(model).__name__} hyperparameter tuning"
            )
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]

            trial_model = copy.deepcopy(model)

            if isinstance(trial_model, XGBoostRegressor):
                kwargs = {
                    "max_depth": trial.suggest_int("max_depth", 1, 5),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 0.8
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-2, 10.0, log=True
                    ),
                    "learning_rate": trial.suggest_categorical(
                        "learning_rate", [0.01, 0.05]
                    ),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "early_stopping_rounds": 10,
                }
                trial_model.xgb_kwargs.update(kwargs)
            elif isinstance(trial_model, SymbolicRegressor):
                trial_model.population_size = trial.suggest_int(
                    "population_size", 50, 500, step=50
                )
                trial_model.generations = trial.suggest_int(
                    "generations", 20, 200, step=10
                )
                trial_model.mutation_rate = trial.suggest_float(
                    "mutation_rate", 0.05, 0.5
                )
                trial_model.crossover_rate = trial.suggest_float(
                    "crossover_rate", 0.4, 0.95
                )
                trial_model.tournament_size = trial.suggest_int(
                    "tournament_size", 2, 10
                )
                trial_model.max_tree_height = trial.suggest_int(
                    "max_tree_height", 2, 12
                )
                trial_model.n_islands = trial.suggest_int(
                    "n_islands", 1, 8
                )
                trial_model.migration_interval = trial.suggest_int(
                    "migration_interval", 2, 20
                )
                trial_model.migration_size = trial.suggest_int(
                    "migration_size", 1, 10
                )
                trial_model.simplify_interval = trial.suggest_int(
                    "simplify_interval", 5, 30, step=5
                )
                trial_model.parsimony_coefficient = trial.suggest_float(
                    "parsimony_coefficient", 0.0001, 1, log=True
                )
                trial_model.basic_arithmetic_only = trial.suggest_categorical(
                    "basic_arithmetic_only", [True, False]
                )
                trial_model.const_opt_top_k_ratio = trial.suggest_float(
                    "const_opt_top_k_ratio", 0.1, 0.5
                )

            elif isinstance(trial_model, ElasticNetRegressor):
                trial_model.alpha = trial.suggest_float("alpha", 0.0001, 0.01, log=True)
                trial_model.l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

            elif isinstance(trial_model, BayesianRidgeRegressor):
                trial_model.alpha_1 = trial.suggest_float(
                    "alpha_1", 1e-6, 0.1, log=True
                )
                trial_model.alpha_2 = trial.suggest_float(
                    "alpha_2", 1e-6, 0.1, log=True
                )
                trial_model.lambda_1 = trial.suggest_float(
                    "lambda_1", 1e-6, 0.1, log=True
                )
                trial_model.lambda_2 = trial.suggest_float(
                    "lambda_2", 1e-6, 0.1, log=True
                )
            elif isinstance(trial_model, HuberRegressor):
                trial_model.epsilon = trial.suggest_float("epsilon", 1, 1000, log=True)
                trial_model.alpha = trial.suggest_float("alpha", 1e-6, 1000, log=True)

            trial_model.fit(
                X_train, y_train, features_name=features_name, eval_set=(X_val, y_val)
            )
            preds = trial_model.predict(X_val)
            score = evaluate_metric(tuning_metric, y_val, preds)
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        for idx, cv in enumerate(cv_scores):
            logger.debug(f"CV score ({tuning_metric}) for fold {idx + 1}: {cv}")
        logger.debug(f"Mean CV score ({tuning_metric}): {mean_score}")
        return mean_score

    logger.debug(
        f"Starting hyperparameter tuning for {type(model).__name__} with {n_trials} trials, optimizing {tuning_metric} ({direction})."
    )

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    for t in study.trials:
        logger.debug(f"Trial {t.number}: {t.params}, value: {t.value}, state: {t.state}")

    logger.debug(f"Best hyperparameters for {type(model).__name__} found: {best_params}, with results ({tuning_metric}) :{study.best_value}")

    best_model = copy.deepcopy(model)

    if isinstance(best_model, XGBoostRegressor):
        best_model.xgb_kwargs.update(best_params)
        best_model.xgb_kwargs["early_stopping_rounds"] = 10
    elif isinstance(best_model, SymbolicRegressor):
        best_model.population_size = best_params["population_size"]
        best_model.generations = best_params["generations"]
        best_model.mutation_rate = best_params["mutation_rate"]
        best_model.crossover_rate = best_params["crossover_rate"]
        best_model.tournament_size = best_params["tournament_size"]
        best_model.max_tree_height = best_params["max_tree_height"]
        best_model.n_islands = best_params["n_islands"]
        best_model.migration_interval = best_params["migration_interval"]
        best_model.migration_size = best_params["migration_size"]
        best_model.simplify_interval = best_params["simplify_interval"]
        best_model.parsimony_coefficient = best_params["parsimony_coefficient"]
        best_model.basic_arithmetic_only = best_params["basic_arithmetic_only"]
        best_model.const_opt_top_k_ratio = best_params["const_opt_top_k_ratio"]
    elif isinstance(best_model, ElasticNetRegressor):
        best_model.alpha = best_params["alpha"]
        best_model.l1_ratio = best_params["l1_ratio"]
    elif isinstance(best_model, BayesianRidgeRegressor):
        best_model.alpha_1 = best_params["alpha_1"]
        best_model.alpha_2 = best_params["alpha_2"]
        best_model.lambda_1 = best_params["lambda_1"]
        best_model.lambda_2 = best_params["lambda_2"]
    elif isinstance(best_model, HuberRegressor):
        best_model.epsilon = best_params["epsilon"]
        best_model.alpha = best_params["alpha"]

    return best_model
