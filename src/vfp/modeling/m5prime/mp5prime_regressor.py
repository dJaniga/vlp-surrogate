import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from vfp.modeling import VFPModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MP5PrimeRegressor(VFPModel):
    _model: DecisionTreeRegressor = field(default=None, init=False)
    _leaf_models_: dict[int, Any] = field(default_factory=dict, init=False)

    max_depth: int = 5
    min_samples_leaf: int = 2
    ccp_alpha: float = 0.0
    leaf_ridge_alpha: float = 1.0
    smoothing_k: float = 15.0
    use_smoothing: bool = True
    use_path_features: bool = True
    seed: int | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> VFPModel:
        self._model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            ccp_alpha=self.ccp_alpha,
            random_state=self.seed,
        )
        self._model.fit(features, targets)

        leaf_ids = self._model.apply(features)
        self._leaf_models_ = {}

        for leaf in np.unique(leaf_ids):
            idx = np.where(leaf_ids == leaf)[0]
            X_leaf = features[idx]
            y_leaf = targets[idx]

            if self.use_path_features:
                path_features = self._get_path_feature_indices(leaf)
                X_leaf = X_leaf[:, path_features]
            else:
                path_features = list(range(features.shape[1]))

            model = make_pipeline(StandardScaler(), Ridge(alpha=self.leaf_ridge_alpha))

            # Ridge requires at least 1 sample; guard against degenerate leaves
            if len(y_leaf) >= 2:
                model.fit(X_leaf, y_leaf)
            else:
                # Fallback: constant predictor wrapped to mimic pipeline.predict
                model = _ConstantPredictor(y_leaf.mean())

            self._leaf_models_[leaf] = (model, path_features)

        logger.debug(
            "MP5PrimeRegressor fitted: %d leaves, tree depth %d",
            len(self._leaf_models_),
            self._model.get_depth(),
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.use_smoothing:
            return self._predict_smoothed(features)
        return self._predict_raw(features)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_raw(self, features: np.ndarray) -> np.ndarray:
        leaf_ids = self._model.apply(features)
        preds = np.zeros(len(features))

        for leaf in np.unique(leaf_ids):
            idx = leaf_ids == leaf
            model, path_features = self._leaf_models_[leaf]
            preds[idx] = model.predict(features[np.ix_(idx, path_features)])

        return preds

    def _predict_smoothed(self, features: np.ndarray) -> np.ndarray:
        """M5' smoothing: blend leaf prediction with ancestor node means up the path."""
        tree = self._model.tree_
        node_indicator = self._model.decision_path(features)
        preds = np.zeros(len(features))

        for i in range(len(features)):
            node_ids = node_indicator[i].indices  # root → leaf path
            leaf_id = node_ids[-1]

            model, path_features = self._leaf_models_[leaf_id]
            prediction = model.predict(features[i, path_features].reshape(1, -1))[0]

            # Smooth bottom-up through ancestors (skip the leaf itself)
            for node_id in reversed(node_ids[:-1]):
                n = tree.n_node_samples[node_id]
                node_mean = tree.value[node_id, 0, 0]
                prediction = (n * prediction + self.smoothing_k * node_mean) / (n + self.smoothing_k)

            preds[i] = prediction

        return preds

    def _get_path_feature_indices(self, leaf_id: int) -> list[int]:
        """Return indices of features used on the decision path to a given leaf."""
        tree = self._model.tree_
        feature_indices: set[int] = set()

        node = 0  # start at root
        while node != leaf_id:
            feat = tree.feature[node]
            threshold = tree.threshold[node]

            # We don't have a sample here, so we reconstruct path by walking the tree
            # and checking whether leaf_id lives in the left or right subtree.
            if self._leaf_in_subtree(tree, tree.children_left[node], leaf_id):
                feature_indices.add(feat)
                node = tree.children_left[node]
            else:
                feature_indices.add(feat)
                node = tree.children_right[node]

        return sorted(feature_indices) if feature_indices else list(range(tree.n_features))

    @staticmethod
    def _leaf_in_subtree(tree, node: int, target: int) -> bool:
        """Check whether `target` node is reachable from `node`."""
        stack = [node]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            left = tree.children_left[current]
            right = tree.children_right[current]
            if left != -1:
                stack.append(left)
            if right != -1:
                stack.append(right)
        return False

    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return "mp5prime_regressor"

    def get_fit_details(self) -> dict[str, Any]:
        if self._model is None:
            return {}
        return {
            "n_leaves": len(self._leaf_models_),
            "tree_depth": self._model.get_depth(),
            "n_training_samples": int(self._model.tree_.n_node_samples[0]),
            "ccp_alpha": self.ccp_alpha,
            "smoothing_k": self.smoothing_k,
            "use_smoothing": self.use_smoothing,
            "use_path_features": self.use_path_features,
        }


class _ConstantPredictor:
    """Minimal fallback predictor for degenerate single-sample leaves."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._value)