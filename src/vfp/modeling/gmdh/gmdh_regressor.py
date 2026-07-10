import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vfp.modeling import VFPModel

from gmdhpy.gmdh import Regressor


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GMDHRegressor(VFPModel):
    _model: Regressor = field(default=None, init=False)
    l2: float = 0.5
    max_layer_count: int = 50
    criterion_minimum_width: int = 5
    layer_err_criterion: str = "top"

    seed: int | None = None

    _normalize: bool = False
    _ref_functions: tuple[str, ...] = ("linear_cov", "linear")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> VFPModel:

        self.features_name = (
            features_name
            if features_name
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        self._model = Regressor(
            normalize=self._normalize,
            ref_functions=self._ref_functions,
            l2=self.l2,
            feature_names=self.features_name,
            max_layer_count=self.max_layer_count,
            criterion_minimum_width=self.criterion_minimum_width,
            layer_err_criterion=self.layer_err_criterion,
            verbose=0,
        )

        self._model.fit(features, targets, validation_data=eval_set)
        logger.debug(f"Model {str(self)} fitted", extra=self.get_fit_details())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def __str__(self) -> str:
        return "gmdh_regressor"

    def get_fit_details(self) -> dict[str, Any]:
        return parse_model_report(self._model.describe())


def parse_model_report(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"model": {}, "layers": []}

    # -------------------------
    # Model section
    # -------------------------
    model_match = re.search(
        r"\*+\s*\nModel\s*\n\*+\s*\n(.*?)(?=\n\*+\s*\nLayer\s+\d+|\Z)",
        text,
        re.DOTALL,
    )

    if model_match:
        model_block = model_match.group(1)

        for line in model_block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = map(str.strip, line.split(":", 1))

            if key == "Number of layers":
                result["model"]["number_of_layers"] = int(value)

            elif key == "Max possible number of layers":
                result["model"]["max_possible_number_of_layers"] = int(value)

            elif key == "Model selection criterion":
                result["model"]["model_selection_criterion"] = value

            elif key == "Number of features":
                result["model"]["number_of_features"] = int(value)

            elif key == "Include features to inputs list for each layer":
                result["model"]["include_features_to_inputs_list_for_each_layer"] = (
                    value.lower() == "true"
                )

            elif key == "Data size":
                result["model"]["data_size"] = int(value)

            elif key == "Train data size":
                result["model"]["train_data_size"] = int(value)

            elif key == "Test data size":
                result["model"]["test_data_size"] = int(value)

            elif key == "Selected features by index":
                result["model"]["selected_features_by_index"] = [
                    int(x) for x in re.findall(r"\d+", value)
                ]

            elif key == "Selected features by name":
                result["model"]["selected_features_by_name"] = [
                    x.strip() for x in value.split(",")
                ]

            elif key == "Unselected features by index":
                result["model"]["unselected_features_by_index"] = [
                    int(x) for x in re.findall(r"\d+", value)
                ]

            elif key == "Unselected features by name":
                result["model"]["unselected_features_by_name"] = (
                    []
                    if value == "No unselected features"
                    else [x.strip() for x in value.split(",")]
                )

    # -------------------------
    # Layers
    # -------------------------
    layer_pattern = re.compile(
        r"\*+\s*\nLayer\s+(\d+)\s*\n\*+\s*\n(.*?)(?=\n\*+\s*\nLayer\s+\d+|\Z)",
        re.DOTALL,
    )

    model_pattern = re.compile(
        r"PolynomModel\s+(\d+)\s*-\s*([^\n]+)\n(.*?)(?=PolynomModel\s+\d+\s*-|\Z)",
        re.DOTALL,
    )

    for layer_match in layer_pattern.finditer(text):
        layer_idx = int(layer_match.group(1))
        layer_block = layer_match.group(2)

        layer_data = {
            "layer_index": layer_idx,
            "models": [],
        }

        for model_match in model_pattern.finditer(layer_block):
            neuron_idx = int(model_match.group(1))
            poly_type = model_match.group(2).strip()
            block = model_match.group(3)

            neuron: dict[str, Any] = {
                "model_index": neuron_idx,
                "polynomial_type": poly_type,
                "inputs": {},
                "weights": {},
            }

            # Inputs
            for input_match in re.finditer(r"(u\d+):\s*(.+)", block):
                neuron["inputs"][input_match.group(1)] = input_match.group(2).strip()

            # Errors
            m = re.search(r"train error:\s*([-\deE.]+)", block)
            if m:
                neuron["train_error"] = float(m.group(1))

            m = re.search(r"validate error:\s*([-\deE.]+)", block)
            if m:
                neuron["validate_error"] = float(m.group(1))

            m = re.search(r"bias error:\s*([-\deE.]+)", block)
            if m:
                neuron["bias_error"] = float(m.group(1))

            # Weights
            for weight_name, weight_value in re.findall(r"(w\d+)=([-\deE.]+)", block):
                neuron["weights"][weight_name] = float(weight_value)

            # Norm
            m = re.search(r"\|\|w\|\|\^2=([-\deE.]+)", block)
            if m:
                neuron["weights_norm_sq"] = float(m.group(1))

            layer_data["models"].append(neuron)

        result["layers"].append(layer_data)

    return result
