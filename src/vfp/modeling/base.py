from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    ) -> VFPModel:
        self.model.fit(features, targets, features_name)

        y_pred = self.predict(features)
        fit_metrics = run_all_regression_metrics(targets, y_pred)

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

        with open(
            Path(self.export_path, f"{str(self.model)}_fit_details").with_suffix(
                ".json"
            ),
            "wb",
        ) as f:
            fit_details = self.model.get_fit_details()
            f.write(orjson.dumps(fit_details, option=orjson.OPT_SERIALIZE_NUMPY))

        df_content = {"T": index.tolist()}
        df_content.update({k: v for k, v in zip(features_name, features.T.tolist())})
        df_content["target"] = targets.flatten().tolist()
        df_content["predicted"] = y_pred.flatten().tolist()
        df = pd.DataFrame(df_content)

        df.to_csv(
            Path(self.export_path, f"{str(self.model)}_fit_data").with_suffix(".csv"),
            index=True,
        )

        return self.model

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
