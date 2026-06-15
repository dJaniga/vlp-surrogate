from __future__ import annotations

import copy
import logging
import numpy as np
import optuna

from sklearn.model_selection import KFold

from vfp.modeling.base import VFPModel
from vfp.modeling.gaussian_process import GaussianProcessRegressor
from vfp.modeling.sklearn_regressors.bayesian_ridge_regressor import (
    BayesianRidgeRegressor,
)
from vfp.modeling.sklearn_regressors.elastic_net_regressor import ElasticNetRegressor
from vfp.modeling.sklearn_regressors.mlp_regressor import MLPRegressor
from vfp.modeling.sklearn_regressors.svr_regressor import SVRRegressor
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
    n_trials: int = 20,
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
            GaussianProcessRegressor,
            SVRRegressor,
            MLPRegressor,
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
                    # Tree structure
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    # Sampling
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "colsample_bylevel": trial.suggest_float(
                        "colsample_bylevel", 0.5, 1.0
                    ),
                    # Regularization
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-8, 10.0, log=True
                    ),
                    # Boosting
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-3, 1, log=True
                    ),
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 100, 1000, step=50
                    ),
                    # Early stopping handled by fit()
                    "early_stopping_rounds": 100,
                }
                trial_model.xgb_kwargs.update(kwargs)
            elif isinstance(trial_model, SymbolicRegressor):
                trial_model.mutation_rate = trial.suggest_float(
                    "mutation_rate", 0.05, 0.7
                )
                trial_model.crossover_rate = trial.suggest_float(
                    "crossover_rate", 0.4, 0.95
                )
                trial_model.tournament_size = trial.suggest_int(
                    "tournament_size", 2, 10
                )
                trial_model.max_tree_height = trial.suggest_int(
                    "max_tree_height", 2, 10
                )
                trial_model.basic_arithmetic_only = trial.suggest_categorical(
                    "basic_arithmetic_only", [True, False]
                )
                trial_model.const_opt_top_k_ratio = trial.suggest_float(
                    "const_opt_top_k_ratio", 0.1, 1.0
                )
                trial_model.parsimony_coefficient = trial.suggest_float(
                    "parsimony_coefficient", 1e-7, 1e-3, log=True
                )

            elif isinstance(trial_model, ElasticNetRegressor):
                trial_model.alpha = trial.suggest_float("alpha", 1e-4, 10, log=True)
                trial_model.l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)

            elif isinstance(trial_model, BayesianRidgeRegressor):
                trial_model.alpha_1 = trial.suggest_float("alpha_1", 1e-6, 1, log=True)
                trial_model.alpha_2 = trial.suggest_float("alpha_2", 1e-6, 1, log=True)
                trial_model.lambda_1 = trial.suggest_float(
                    "lambda_1", 1e-6, 1, log=True
                )
                trial_model.lambda_2 = trial.suggest_float(
                    "lambda_2", 1e-6, 1, log=True
                )
            elif isinstance(trial_model, HuberRegressor):
                trial_model.epsilon = trial.suggest_float("epsilon", 1.01, 1000)
                trial_model.alpha = trial.suggest_float("alpha", 1e-4, 10, log=True)
            elif isinstance(trial_model, GaussianProcessRegressor):
                trial_model.kernel_name = trial.suggest_categorical(
                    "kernel_name",
                    ["rbf", "ard", "matern52", "polynomial", "rational_quadratic"],
                )
                trial_model.noise_variance = trial.suggest_float(
                    "noise_variance", 1e-6, 10, log=True
                )
                if trial_model.kernel_name == "polynomial":
                    trial_model.degree = trial.suggest_int("degree", 1, 4)

            elif isinstance(trial_model, SVRRegressor):
                trial_model.C = trial.suggest_float("C", 1e-6, 10, log=True)
                trial_model.epsilon = trial.suggest_float("epsilon", 1e-6, 10, log=True)
                trial_model.kernel = trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                )
                trial_model.degree = trial.suggest_int("degree", 1, 5)
            elif isinstance(trial_model, MLPRegressor):
                n_layers = trial.suggest_int("n_layers", 1, 3)
                hidden_layer_sizes = tuple(
                    trial.suggest_int(f"n_units_l{i}", 32, 256, step=32)
                    for i in range(n_layers)
                )
                trial_model.hidden_layer_sizes = hidden_layer_sizes
                trial_model.activation = trial.suggest_categorical(
                    "activation", ["relu", "tanh", "logistic"]
                )
                trial_model.alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
                trial_model.learning_rate = trial.suggest_categorical(
                    "learning_rate", ["constant", "invscaling", "adaptive"]
                )
                trial_model.learning_rate_init = trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-1, log=True
                )
                trial_model.beta_1 = trial.suggest_float("beta_1", 0.85, 0.99)
                trial_model.beta_2 = trial.suggest_float("beta_2", 0.99, 0.9999)
                trial_model.max_iter = trial.suggest_int("max_iter", 100, 500, step=100)
                trial_model.n_iter_no_change = trial.suggest_int(
                    "n_iter_no_change", 5, 20
                )

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
        logger.debug(
            f"Trial {t.number}: {t.params}, value: {t.value}, state: {t.state}"
        )

    logger.debug(
        f"Best hyperparameters for {type(model).__name__} found: {best_params}, with results ({tuning_metric}) :{study.best_value}"
    )

    best_model = copy.deepcopy(model)

    if isinstance(best_model, XGBoostRegressor):
        best_model.xgb_kwargs.update(best_params)
        best_model.xgb_kwargs["early_stopping_rounds"] = 100
    elif isinstance(best_model, SymbolicRegressor):
        best_model.mutation_rate = best_params["mutation_rate"]
        best_model.crossover_rate = best_params["crossover_rate"]
        best_model.tournament_size = best_params["tournament_size"]
        best_model.max_tree_height = best_params["max_tree_height"]
        best_model.basic_arithmetic_only = best_params["basic_arithmetic_only"]
        best_model.const_opt_top_k_ratio = best_params["const_opt_top_k_ratio"]
        best_model.parsimony_coefficient = best_params["parsimony_coefficient"]
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
    elif isinstance(best_model, GaussianProcessRegressor):
        best_model.kernel_name = best_params["kernel_name"]
        best_model.noise_variance = best_params["noise_variance"]
        if best_model.kernel_name == "polynomial":
            best_model.degree = best_params["degree"]
    elif isinstance(best_model, SVRRegressor):
        best_model.C = best_params["C"]
        best_model.epsilon = best_params["epsilon"]
        best_model.kernel = best_params["kernel"]
        best_model.degree = best_params["degree"]
    elif isinstance(best_model, MLPRegressor):
        best_model.hidden_layer_sizes = tuple(
            best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])
        )
        best_model.activation = best_params["activation"]
        best_model.alpha = best_params["alpha"]
        best_model.learning_rate = best_params["learning_rate"]
        best_model.learning_rate_init = best_params["learning_rate_init"]
        best_model.beta_1 = best_params["beta_1"]
        best_model.beta_2 = best_params["beta_2"]
        best_model.max_iter = best_params["max_iter"]
        best_model.n_iter_no_change = best_params["n_iter_no_change"]

    return best_model
