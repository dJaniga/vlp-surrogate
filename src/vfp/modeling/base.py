from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split, KFold
import copy
import numpy as np
import orjson
import pandas as pd

from toolbox import run_all_regression_metrics

logger = logging.getLogger(__name__)


class VFPModel(ABC):
    def __init__(self) -> None:
        self.features_name: tuple[str, ...] | None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> VFPModel: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def get_fit_details(self) -> dict[str, Any]: ...


@dataclass
class ModelWrapper:
    model: VFPModel
    export_path: Path

    def fit(
        self,
        index: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        optimize_hyperparameters: bool = False,
        tuning_metric: str = "mean_squared_error",
    ) -> VFPModel:
        # Determine features names safely
        _features_name: tuple[str, ...] = (
            features_name
            if features_name is not None
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        # Hyperparameter Tuning using Optuna
        if optimize_hyperparameters:
            from vfp.modeling.tuning import tune_hyperparameters

            self.model = tune_hyperparameters(
                self.model, features, targets, tuning_metric=tuning_metric
            )

        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        # Ensure typed ndarrays
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        self.model.fit(
            X_train, y_train, features_name=_features_name, eval_set=(X_test, y_test)
        )

        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        # compute k-Fold Cross-Validation metrics
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics_list = []
        for idx, (train_idx, val_idx) in enumerate(kf.split(features)):
            logger.info(f"Running CV fold {idx+1} of 5")
            X_cv_train, X_cv_val = features[train_idx], features[val_idx]
            y_cv_train, y_cv_val = targets[train_idx], targets[val_idx]
            cv_model = copy.deepcopy(self.model)
            cv_model.fit(
                X_cv_train,
                y_cv_train,
                features_name=_features_name,
                eval_set=(X_cv_val, y_cv_val),
            )
            y_cv_pred = cv_model.predict(X_cv_val)
            cv_metrics_list.append(run_all_regression_metrics(y_cv_val, y_cv_pred))

        cv_metrics = {}
        if cv_metrics_list:
            for key in cv_metrics_list[0].keys():
                cv_metrics[key] = float(
                    np.mean(
                        [
                            m[key]
                            for m in cv_metrics_list
                            if m[key] is not None and not np.isnan(m[key])
                        ]
                    )
                )

        fit_metrics = {
            "train": run_all_regression_metrics(y_train, y_pred_train),
            "test": run_all_regression_metrics(y_test, y_pred_test),
            "cv_validation": cv_metrics,
        }

        logger.info(
            "Fit diagnostics",
            extra={"fit_metrics": fit_metrics},
        )

        self.export_path.mkdir(parents=True, exist_ok=True)

        with open(
            Path(self.export_path, f"{str(self.model)}_fit_results").with_suffix(
                ".json"
            ),
            "w",
        ) as f:
            json.dump(fit_metrics, f, indent=4)

        # Aggregate metrics table and save it
        metrics_df = (
            pd.DataFrame(fit_metrics).reset_index().rename(columns={"index": "Metric"})
        )
        metrics_df.to_csv(
            Path(self.export_path, f"{str(self.model)}_fit_results").with_suffix(
                ".csv"
            ),
            index=False,
        )

        with open(
            Path(self.export_path, f"{str(self.model)}_fit_details").with_suffix(
                ".json"
            ),
            "wb",
        ) as f:
            fit_details = self.model.get_fit_details()
            f.write(orjson.dumps(fit_details, option=orjson.OPT_SERIALIZE_NUMPY))

        # Re-predict on all features for the CSV
        y_pred_all = self.predict(features)
        df_content = {"T": index.tolist()}
        df_content.update({k: v for k, v in zip(_features_name, features.T.tolist())})
        df_content["target"] = targets.flatten().tolist()
        df_content["predicted"] = y_pred_all.flatten().tolist()
        df = pd.DataFrame(df_content)

        df.to_csv(
            Path(self.export_path, f"{str(self.model)}_fit_data").with_suffix(".csv"),
            index=True,
        )

        return self.model

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
