from __future__ import annotations

import copy
import logging
import numpy as np
import optuna
from sklearn.model_selection import KFold

from vfp.modeling.base import VFPModel
from vfp.modeling.xgboost import XGBoostRegressor
from vfp.modeling.symbolic.regressor import SymbolicRegressor
from vfp.modeling.tuning_metrics import evaluate_metric, get_metric_direction

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    model: VFPModel,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: tuple[str, ...],
    n_trials: int = 20,
    cv_folds: int = 5,
    tuning_metric: str = "mean_squared_error",
    seed: int | None = None,
) -> VFPModel:
    if not isinstance(model, (XGBoostRegressor, SymbolicRegressor)):
        logger.info(
            f"Hyperparameter tuning not implemented for {type(model).__name__}. Returning original model."
        )
        return model

    def objective(trial: optuna.Trial) -> float:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores = []

        for idx, (train_idx, val_idx) in enumerate(kf.split(features)):
            logger.info(
                f"Running CV fold {idx + 1} of {cv_folds} for {type(model).__name__} hyperparameter tuning"
            )
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]

            trial_model = copy.deepcopy(model)

            if isinstance(trial_model, XGBoostRegressor):
                kwargs = {
                    "max_depth": trial.suggest_int("max_depth", 1, 3),
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
                trial_model.parsimony_coefficient = trial.suggest_float(
                    "parsimony_coefficient", 0.0001, 0.01, log=True
                )
                trial_model.basic_arithmetic_only = True

            trial_model.fit(
                X_train, y_train, features_name=features_name, eval_set=(X_val, y_val)
            )
            preds = trial_model.predict(X_val)
            cv_scores.append(evaluate_metric(tuning_metric, y_val, preds))

        return float(np.mean(cv_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = get_metric_direction(tuning_metric)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    logger.info(
        f"Starting hyperparameter tuning for {type(model).__name__} with {n_trials} trials, optimizing {tuning_metric} ({direction})."
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best hyperparameters found: {best_params}")

    best_model = copy.deepcopy(model)
    if isinstance(best_model, XGBoostRegressor):
        best_model.xgb_kwargs.update(best_params)
    elif isinstance(best_model, SymbolicRegressor):
        best_model.parsimony_coefficient = best_params["parsimony_coefficient"]
        best_model.basic_arithmetic_only = True

    return best_model
