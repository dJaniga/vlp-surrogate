from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
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


def compute_cycle_ids(dates: np.ndarray, cycle_gap_days: int) -> np.ndarray:
    """
    Assigns a cycle ID to each observation based on elapsed time between
    consecutive (chronologically sorted) observations.

    A new cycle starts whenever the gap between two consecutive dates
    exceeds `cycle_gap_days` (e.g. a switch from injection back to
    production after a shut-in/injection period).

    Parameters
    ----------
    dates : np.ndarray
        Array of datetime64 values, assumed sorted ascending.
    cycle_gap_days : int
        Threshold, in days, above which a gap is considered a new cycle.

    Returns
    -------
    np.ndarray
        Integer array of the same length as `dates`, with a cycle ID
        (0-indexed, monotonically increasing) for each observation.
    """
    if len(dates) == 0:
        return np.array([], dtype=int)

    dt_days = np.diff(dates).astype("timedelta64[D]").astype(int)
    cycle_breaks = dt_days > cycle_gap_days
    cycle_id = np.concatenate([[0], np.cumsum(cycle_breaks)])
    return cycle_id.astype(int)


@dataclass
class ModelWrapper:
    model: VFPModel
    export_path: Path
    seed: int | None = None
    feature_scaler: StandardScaler = field(default_factory=StandardScaler)
    target_scaler: StandardScaler = field(default_factory=StandardScaler)
    group_by_cycle: bool = True
    cycle_gap_days: int = 100

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

        logger.info(f">>>> Starting fitting model {self.model}...")

        _features_name: tuple[str, ...] = (
            features_name
            if features_name is not None
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        features = np.asarray(features)
        targets = np.asarray(targets)

        # ------------------------------------------------------------------ #
        # Ensure chronological order before computing gaps / cycle IDs.       #
        # Downstream code (dt, cycle_id) assumes `dates` is sorted ascending, #
        # so we sort everything consistently by date first.                  #
        # ------------------------------------------------------------------ #
        dates = index.astype("datetime64[D]")
        sort_order = np.argsort(dates)
        if not np.array_equal(sort_order, np.arange(len(dates))):
            logger.warning(
                "Input index was not sorted chronologically — sorting "
                "features/targets/index by date before cycle detection."
            )
        dates = dates[sort_order]
        index = index[sort_order]
        features = features[sort_order]
        targets = targets[sort_order]

        dt = np.diff(dates).astype("timedelta64[D]")

        # ------------------------------------------------------------------ #
        # 0. STANDARDIZE features and targets                                  #
        # ------------------------------------------------------------------ #
        features = self.feature_scaler.fit_transform(features)
        targets = self.target_scaler.fit_transform(targets.reshape(-1, 1)).ravel()

        # ------------------------------------------------------------------ #
        # 1. NESTED CROSS-VALIDATION — unbiased generalization estimate        #
        #    Outer fold: held-out eval                                         #
        #    Inner fold (optional): hyperparameter tuning                      #
        #                                                                       #
        #    If group_by_cycle is True, entire production cycles are kept      #
        #    together in either train or validation — never split across       #
        #    both — to avoid leakage from within-cycle autocorrelation.        #
        #    Cycles are detected as runs of observations separated by gaps     #
        #    of at most `cycle_gap_days`; a gap larger than that (e.g. an      #
        #    injection period) marks the start of a new cycle.                 #
        # ------------------------------------------------------------------ #
        if self.group_by_cycle:
            cycle_id = compute_cycle_ids(dates, self.cycle_gap_days)
            n_cycles = len(np.unique(cycle_id))
            logger.info(
                f">>>> Detected {n_cycles} production cycle(s) "
                f"(gap threshold: {self.cycle_gap_days} days)."
            )

            if n_cycles < 2:
                logger.warning(
                    "Only one cycle detected — group-based CV cannot create "
                    "meaningful splits. Falling back to standard shuffled "
                    "KFold. Check cycle_gap_days or your date index if this "
                    "is unexpected."
                )
                outer_kf = KFold(
                    n_splits=outer_splits, shuffle=True, random_state=self.seed
                )
                split_iter = outer_kf.split(features)
            else:
                effective_splits = min(outer_splits, n_cycles)
                if effective_splits < outer_splits:
                    logger.warning(
                        f"Requested outer_splits={outer_splits} but only "
                        f"{n_cycles} cycles are available — reducing to "
                        f"{effective_splits} splits so every fold gets at "
                        f"least one full cycle."
                    )
                outer_kf = GroupKFold(n_splits=effective_splits)
                split_iter = outer_kf.split(features, groups=cycle_id)
        else:
            cycle_id = None
            outer_kf = KFold(
                n_splits=outer_splits, shuffle=True, random_state=self.seed
            )
            split_iter = outer_kf.split(features)

        outer_metrics_list: list[dict] = []
        outer_fold_sizes: list[int] = []

        for outer_idx, (outer_train_idx, outer_val_idx) in enumerate(split_iter):
            logger.debug(f"*** Outer CV fold {outer_idx + 1} ***")

            X_outer_train, X_outer_val = (
                features[outer_train_idx],
                features[outer_val_idx],
            )
            y_outer_train, y_outer_val = (
                targets[outer_train_idx],
                targets[outer_val_idx],
            )

            fold_model = copy.deepcopy(self.model)

            # Inner loop: tune only on outer training data, never touches outer_val.
            # Cycle groups (if enabled) are sliced to the outer-train subset so
            # the inner CV used for hyperparameter search also respects cycle
            # boundaries, consistent with the outer CV split above.
            if optimize_hyperparameters:
                from vfp.modeling.tuning import tune_hyperparameters

                inner_groups = (
                    cycle_id[outer_train_idx] if cycle_id is not None else None
                )

                fold_model = tune_hyperparameters(
                    fold_model,
                    X_outer_train,
                    y_outer_train,
                    tuning_metric=tuning_metric,
                    features_name=_features_name,
                    seed=self.seed,
                    n_splits=inner_splits,  # inner CV splits
                    groups=inner_groups,
                )

            fold_model.fit(
                X_outer_train,
                y_outer_train,
                features_name=_features_name,
                eval_set=(X_outer_val, y_outer_val),
            )

            y_outer_pred = fold_model.predict(X_outer_val)
            y_outer_val_orig = self.target_scaler.inverse_transform(
                y_outer_val.reshape(-1, 1)
            ).ravel()
            y_outer_pred_orig = self.target_scaler.inverse_transform(
                y_outer_pred.reshape(-1, 1)
            ).ravel()
            outer_metrics_list.append(
                run_all_regression_metrics(y_outer_val_orig, y_outer_pred_orig)
            )
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

        logger.debug(
            "*** Nested CV complete ***", extra={"nested_cv_metrics": nested_cv_metrics}
        )

        # ------------------------------------------------------------------ #
        # 2. FINAL MODEL — trained on ALL data                                #
        #    Hyperparameters tuned on all data via inner CV (no holdout leak) #
        # ------------------------------------------------------------------ #
        logger.debug("*** Training final model on full dataset ***")

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
                groups=cycle_id,
            )

        self.model.fit(features, targets, features_name=_features_name)

        # Training-set metrics (expected to be optimistic — for diagnostics only)
        y_pred_train_all = self.predict(self.feature_scaler.inverse_transform(features))
        targets_orig = self.target_scaler.inverse_transform(
            targets.reshape(-1, 1)
        ).ravel()
        train_metrics = run_all_regression_metrics(targets_orig, y_pred_train_all)

        fit_metrics = {
            "full_dataset": train_metrics,  # optimistic, diagnostic only
            "cv_datasets": nested_cv_metrics,  # unbiased generalization estimate
        }

        logger.debug("Fit diagnostics", extra={"fit_metrics": fit_metrics})

        logger.info(
            f">>>> Model {self.model} fit completed with target metric {tuning_metric}: {train_metrics[tuning_metric]}"
        )

        # ------------------------------------------------------------------ #
        # 3. EXPORT                                                            #
        # ------------------------------------------------------------------ #
        self.export_path.mkdir(parents=True, exist_ok=True)
        well_name = self.export_path.parts[-1]

        with open(
            Path(
                self.export_path,
                f"{well_name}_{str(self.model)}_{tuning_metric}_fit_results",
            ).with_suffix(".json"),
            "w",
        ) as f:
            json.dump(fit_metrics, f, indent=4)

        metrics_df = (
            pd.DataFrame(fit_metrics).reset_index().rename(columns={"index": "Metric"})
        )
        metrics_df.to_csv(
            Path(
                self.export_path,
                f"{well_name}_{str(self.model)}_{tuning_metric}_fit_results",
            ).with_suffix(".csv"),
            index=False,
        )

        with open(
            Path(
                self.export_path,
                f"{well_name}_{str(self.model)}_{tuning_metric}_fit_details",
            ).with_suffix(".json"),
            "wb",
        ) as f:
            f.write(
                orjson.dumps(
                    self.model.get_fit_details(), option=orjson.OPT_SERIALIZE_NUMPY
                )
            )

        # Export predictions from the final model on full data
        features_orig = self.feature_scaler.inverse_transform(features)
        y_pred_all = self.predict(features_orig)
        df_content = {"T": index.tolist()}
        df_content.update(
            {k: v for k, v in zip(_features_name, features_orig.T.tolist())}
        )
        df_content["target"] = targets_orig.flatten().tolist()
        df_content["predicted"] = y_pred_all.flatten().tolist()

        pd.DataFrame(df_content).to_csv(
            Path(
                self.export_path,
                f"{well_name}_{str(self.model)}_{tuning_metric}_fit_data",
            ).with_suffix(".csv"),
            index=True,
        )

        return self.model

    def predict(self, features: np.ndarray) -> np.ndarray:
        features_scaled = self.feature_scaler.transform(np.asarray(features))
        predictions_scaled = self.model.predict(features_scaled)
        return self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).ravel()