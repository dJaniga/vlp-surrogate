import logging
import os
from dataclasses import dataclass, field
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

from vfp.modeling import VFPModel
from sklearn.neural_network import MLPRegressor as SKMLPRegressor

from vfp.modeling.tuning_metrics import evaluate_metric

logger = logging.getLogger(__name__)

_FIT_METRIC: str = os.environ.get("VLP_FIT_METRIC", "mse")


@dataclass(slots=True)
class MLPRegressor(VFPModel):
    _model: SKMLPRegressor = field(default=None, init=False)
    hidden_layer_sizes = (100,)
    activation = "relu"
    alpha = 0.0001
    solver = "lbfgs"
    learning_rate = "invscaling"
    learning_rate_init = 0.001
    max_iter = 500
    shuffle = True
    beta_1 = 0.9
    beta_2 = 0.999
    n_iter_no_change = 50
    seed: int | None = None
    _early_stopping: bool = False
    _eval_history: list[dict[str, float]] = field(default_factory=list)
    _tol: float = 1e-6

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> VFPModel:

        n_iter = self.max_iter

        self._model = SKMLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            shuffle=self.shuffle,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            random_state=self.seed,
            early_stopping=False,
            warm_start=True,
        )

        best_eval_loss: float = float("inf")
        no_improvement_count: int = 0
        self._eval_history: list[dict[str, float]] = []

        for iteration in range(1, n_iter + 1):
            self._model.partial_fit(features, targets)

            train_loss = evaluate_metric(
                _FIT_METRIC, targets, self._model.predict(features)
            )
            iteration_log: dict[str, float] = {
                "iteration": iteration,
                f"train_{_FIT_METRIC}": train_loss,
            }

            if eval_set is not None:
                eval_features, eval_targets = eval_set
                eval_preds = self._model.predict(eval_features)
                eval_loss = evaluate_metric(_FIT_METRIC, eval_targets, eval_preds)
                iteration_log[f"eval_{_FIT_METRIC}"] = eval_loss
                iteration_log[f"best_eval_{_FIT_METRIC}"] = best_eval_loss
                monitored_loss = eval_loss
            else:
                monitored_loss = train_loss

            self._eval_history.append(iteration_log)
            # logger.debug("Iter %d | %s", iteration, iteration_log)

            if monitored_loss < best_eval_loss - self._tol:
                best_eval_loss = monitored_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.n_iter_no_change:
                    logger.info(
                        "Early stop at iteration %d — no improvement for %d iterations.",
                        iteration,
                        self.n_iter_no_change,
                    )
                    break

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def __str__(self) -> str:
        return "mlp_regressor"

    def get_fit_details(self) -> dict[str, Any]:
        return {}
