from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.model_selection import KFold
import copy
import numpy as np
import orjson
import pandas as pd

from toolbox import run_all_regression_metrics

logger = logging.getLogger(__name__)


class VFPModel(ABC):
    def __init__(self) -> None:
        self.features_name: tuple[str, ...] | None

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> VFPModel: ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def get_fit_details(self) -> dict[str, Any]: ...


@dataclass
class ModelWrapper:
    model: VFPModel
    export_path: Path
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed is None:
            self.seed = getattr(self.model, "seed", None)

    def fit(
        self,
        index: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        optimize_hyperparameters: bool = False,
        tuning_metric: str = "mean_squared_error",
        outer_splits: int = 5,
        inner_splits: int = 5,
    ) -> VFPModel:
        _features_name: tuple[str, ...] = (
            features_name
            if features_name is not None
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        features = np.asarray(features)
        targets = np.asarray(targets)

        # ------------------------------------------------------------------ #
        # 1. NESTED CROSS-VALIDATION — unbiased generalization estimate        #
        #    Outer fold: held-out eval                                         #
        #    Inner fold (optional): hyperparameter tuning                      #
        # ------------------------------------------------------------------ #
        outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=self.seed)
        outer_metrics_list: list[dict] = []
        outer_fold_sizes: list[int] = []

        for outer_idx, (outer_train_idx, outer_val_idx) in enumerate(outer_kf.split(features)):
            logger.info(f"Outer CV fold {outer_idx + 1}/{outer_splits}")

            X_outer_train, X_outer_val = features[outer_train_idx], features[outer_val_idx]
            y_outer_train, y_outer_val = targets[outer_train_idx], targets[outer_val_idx]

            fold_model = copy.deepcopy(self.model)

            # Inner loop: tune only on outer training data, never touches outer_val
            if optimize_hyperparameters:
                from vfp.modeling.tuning import tune_hyperparameters

                fold_model = tune_hyperparameters(
                    fold_model,
                    X_outer_train,
                    y_outer_train,
                    tuning_metric=tuning_metric,
                    features_name=_features_name,
                    seed=self.seed,
                    n_splits=inner_splits,  # inner CV splits
                )

            fold_model.fit(
                X_outer_train,
                y_outer_train,
                features_name=_features_name,
                eval_set=(X_outer_val, y_outer_val),
            )

            y_outer_pred = fold_model.predict(X_outer_val)
            outer_metrics_list.append(run_all_regression_metrics(y_outer_val, y_outer_pred))
            outer_fold_sizes.append(len(outer_val_idx))

        # Weighted average across outer folds (accounts for unequal fold sizes)
        nested_cv_metrics: dict[str, float] = {}
        if outer_metrics_list:
            for key in outer_metrics_list[0].keys():
                values_and_weights = [
                    (m[key], outer_fold_sizes[i])
                    for i, m in enumerate(outer_metrics_list)
                    if m[key] is not None and not np.isnan(m[key])
                ]
                if values_and_weights:
                    scores, weights = zip(*values_and_weights)
                    nested_cv_metrics[key] = float(np.average(scores, weights=weights))

        logger.info("Nested CV complete", extra={"nested_cv_metrics": nested_cv_metrics})

        # ------------------------------------------------------------------ #
        # 2. FINAL MODEL — trained on ALL data                                #
        #    Hyperparameters tuned on all data via inner CV (no holdout leak) #
        # ------------------------------------------------------------------ #
        logger.info("Training final model on full dataset")

        if optimize_hyperparameters:
            from vfp.modeling.tuning import tune_hyperparameters

            self.model = tune_hyperparameters(
                self.model,
                features,
                targets,
                tuning_metric=tuning_metric,
                features_name=_features_name,
                seed=self.seed,
                n_splits=inner_splits,
            )

        self.model.fit(features, targets, features_name=_features_name)

        # Training-set metrics (expected to be optimistic — for diagnostics only)
        y_pred_train_all = self.predict(features)
        train_metrics = run_all_regression_metrics(targets, y_pred_train_all)

        fit_metrics = {
            "train_resubstitution": train_metrics,   # optimistic, diagnostic only
            "nested_cv": nested_cv_metrics,          # unbiased generalization estimate
        }

        logger.info("Fit diagnostics", extra={"fit_metrics": fit_metrics})

        # ------------------------------------------------------------------ #
        # 3. EXPORT                                                            #
        # ------------------------------------------------------------------ #
        self.export_path.mkdir(parents=True, exist_ok=True)

        with open(
            Path(self.export_path, f"{str(self.model)}_fit_results").with_suffix(".json"), "w"
        ) as f:
            json.dump(fit_metrics, f, indent=4)

        metrics_df = (
            pd.DataFrame(fit_metrics).reset_index().rename(columns={"index": "Metric"})
        )
        metrics_df.to_csv(
            Path(self.export_path, f"{str(self.model)}_fit_results").with_suffix(".csv"),
            index=False,
        )

        with open(
            Path(self.export_path, f"{str(self.model)}_fit_details").with_suffix(".json"), "wb"
        ) as f:
            f.write(orjson.dumps(self.model.get_fit_details(), option=orjson.OPT_SERIALIZE_NUMPY))

        # Export predictions from the final model on full data
        y_pred_all = self.predict(features)
        df_content = {"T": index.tolist()}
        df_content.update({k: v for k, v in zip(_features_name, features.T.tolist())})
        df_content["target"] = targets.flatten().tolist()
        df_content["predicted"] = y_pred_all.flatten().tolist()

        pd.DataFrame(df_content).to_csv(
            Path(self.export_path, f"{str(self.model)}_fit_data").with_suffix(".csv"),
            index=True,
        )

        return self.model

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)