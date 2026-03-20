from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from tools.ml.model_factory import ModelFactory


def _rank_features_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    random_state: int,
) -> np.ndarray:
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": int(random_state),
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = ModelFactory.criar_lightgbm(task=task, params=params)
    model.fit(X_train, y_train)

    importancias = np.asarray(
        getattr(model, "feature_importances_", np.zeros(X_train.shape[1])),
        dtype=float,
    )
    if importancias.size != X_train.shape[1] or np.allclose(importancias, 0.0):
        importancias = np.nan_to_num(np.var(X_train, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    return np.argsort(-importancias)


class FeaturePostProcessor:
    def __init__(self) -> None:
        self.upstream: Optional["FeaturePostProcessor"] = None

        self.base_feature_names: List[str] = []
        self.selected_feature_indices: List[int] = []
        self.feature_names_after_selection: List[str] = []

        self.reduction_enabled = False
        self.reduction_keep_indices: List[int] = []
        self.reduction_tail_indices: List[int] = []
        self.reduction_tail_feature_names: List[str] = []
        self.reduction_component_names: List[str] = []
        self.reduction_imputer: Optional[SimpleImputer] = None
        self.reduction_model: Optional[PCA] = None
        self.feature_names_after_reduction: List[str] = []
        self.reduction_metadata: Dict[str, Any] = {}

        self.final_selected_feature_indices: List[int] = []
        self.feature_names_out: List[str] = []

    def clone(self) -> "FeaturePostProcessor":
        return deepcopy(self)

    def _resolve_input_space(
        self,
        X_train: np.ndarray,
        feature_names: Sequence[str],
        upstream: Optional["FeaturePostProcessor"] = None,
    ) -> np.ndarray:
        if upstream is not None:
            self.upstream = upstream.clone()
            X_base = self.upstream.transform(np.asarray(X_train))
            self.base_feature_names = list(self.upstream.feature_names_out)
            return X_base

        self.upstream = None
        X_base = np.asarray(X_train)
        n_features = int(X_base.shape[1])
        self.base_feature_names = (
            list(feature_names)
            if len(feature_names) == n_features
            else [f"f_{i}" for i in range(n_features)]
        )
        return X_base

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Sequence[str],
        selected_feature_indices: Optional[Sequence[int]] = None,
        task: str = "classification",
        reduction_enabled: bool = False,
        reduction_tail_features: int = 0,
        reduction_tail_indices: Optional[Sequence[int]] = None,
        reduction_variance_threshold: float = 0.70,
        random_state: int = 42,
        upstream: Optional["FeaturePostProcessor"] = None,
    ) -> np.ndarray:
        X_base = self._resolve_input_space(
            X_train=np.asarray(X_train),
            feature_names=feature_names,
            upstream=upstream,
        )
        n_features = int(X_base.shape[1])

        if selected_feature_indices is None:
            self.selected_feature_indices = list(range(n_features))
        else:
            self.selected_feature_indices = [int(idx) for idx in selected_feature_indices]

        X_selected = X_base[:, self.selected_feature_indices]
        self.feature_names_after_selection = [
            self.base_feature_names[idx] for idx in self.selected_feature_indices
        ]

        X_after_reduction = X_selected
        self.feature_names_after_reduction = list(self.feature_names_after_selection)
        self.reduction_enabled = False
        self.reduction_keep_indices = []
        self.reduction_tail_indices = []
        self.reduction_tail_feature_names = []
        self.reduction_component_names = []
        self.reduction_imputer = None
        self.reduction_model = None
        self.reduction_metadata = {
            "enabled": False,
            "requested_tail_features": int(reduction_tail_features or 0),
            "requested_tail_indices": [int(idx) for idx in (reduction_tail_indices or [])],
            "variance_threshold": float(reduction_variance_threshold or 0.70),
        }

        tail_indices_local = (
            [int(idx) for idx in reduction_tail_indices]
            if reduction_tail_indices is not None
            else []
        )
        tail_features = int(reduction_tail_features or 0)
        variance_threshold = float(reduction_variance_threshold or 0.70)

        if reduction_enabled and X_selected.shape[1] >= 3:
            if not tail_indices_local and tail_features >= 2:
                tail_features = min(tail_features, max(0, X_selected.shape[1] - 1))
                if tail_features >= 2:
                    ranked_idx = _rank_features_lightgbm(
                        X_train=X_selected,
                        y_train=np.asarray(y_train),
                        task=task,
                        random_state=random_state,
                    )
                    tail_indices_local = sorted([int(i) for i in ranked_idx[-tail_features:]])

            tail_indices_local = sorted(
                idx for idx in tail_indices_local if 0 <= int(idx) < X_selected.shape[1]
            )

            if tail_indices_local and len(tail_indices_local) < X_selected.shape[1]:
                tail_set = set(tail_indices_local)
                keep_idx = [i for i in range(X_selected.shape[1]) if i not in tail_set]

                tail_train = X_selected[:, tail_indices_local]
                imputer = SimpleImputer(strategy="median")
                tail_train = imputer.fit_transform(tail_train)

                pca_full = PCA(random_state=random_state)
                pca_full.fit(tail_train)
                explained_ratio = np.asarray(
                    getattr(pca_full, "explained_variance_ratio_", []),
                    dtype=float,
                )

                if explained_ratio.size:
                    cumulative = np.cumsum(explained_ratio)
                    n_components = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
                    n_components = max(1, min(n_components, len(tail_indices_local)))

                    self.reduction_enabled = True
                    self.reduction_keep_indices = keep_idx
                    self.reduction_tail_indices = tail_indices_local
                    self.reduction_tail_feature_names = [
                        self.feature_names_after_selection[i] for i in tail_indices_local
                    ]
                    self.reduction_component_names = [
                        f"pca_tail_{i + 1}" for i in range(n_components)
                    ]
                    self.reduction_imputer = imputer
                    self.reduction_model = PCA(
                        n_components=n_components,
                        random_state=random_state,
                    )

                    tail_reduced = self.reduction_model.fit_transform(tail_train)
                    X_after_reduction = np.hstack([X_selected[:, keep_idx], tail_reduced])
                    self.feature_names_after_reduction = (
                        [self.feature_names_after_selection[i] for i in keep_idx]
                        + list(self.reduction_component_names)
                    )
                    self.reduction_metadata = {
                        "enabled": True,
                        "method": "pca_tail_features",
                        "requested_tail_features": int(reduction_tail_features or 0),
                        "requested_tail_indices": [int(idx) for idx in tail_indices_local],
                        "variance_threshold": variance_threshold,
                        "tail_features_used": len(tail_indices_local),
                        "n_components_applied": int(n_components),
                        "tail_feature_names": list(self.reduction_tail_feature_names),
                        "explained_variance_ratio": [
                            float(v) for v in self.reduction_model.explained_variance_ratio_.tolist()
                        ],
                        "cumulative_explained_variance": [
                            float(v)
                            for v in np.cumsum(self.reduction_model.explained_variance_ratio_).tolist()
                        ],
                    }

        self.final_selected_feature_indices = list(range(X_after_reduction.shape[1]))
        self.feature_names_out = list(self.feature_names_after_reduction)
        return X_after_reduction

    def transform_base(self, X: np.ndarray) -> np.ndarray:
        X_base = np.asarray(X)
        if self.upstream is not None:
            X_base = self.upstream.transform(X_base)

        X_selected = X_base[:, self.selected_feature_indices]

        if not self.reduction_enabled:
            return X_selected

        keep_part = X_selected[:, self.reduction_keep_indices]
        tail_part = X_selected[:, self.reduction_tail_indices]
        tail_part = self.reduction_imputer.transform(tail_part)
        tail_reduced = self.reduction_model.transform(tail_part)
        return np.hstack([keep_part, tail_reduced])

    def set_final_feature_indices(self, feature_indices: Optional[Sequence[int]]) -> None:
        if feature_indices is None:
            self.final_selected_feature_indices = list(
                range(len(self.feature_names_after_reduction))
            )
        else:
            self.final_selected_feature_indices = [int(idx) for idx in feature_indices]

        self.feature_names_out = [
            self.feature_names_after_reduction[idx]
            for idx in self.final_selected_feature_indices
        ]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_base = self.transform_base(X)
        if not self.final_selected_feature_indices:
            return X_base
        return X_base[:, self.final_selected_feature_indices]

    def metadata(self) -> Dict[str, Any]:
        return {
            "upstream": self.upstream.metadata() if self.upstream is not None else None,
            "selected_feature_indices": list(self.selected_feature_indices),
            "feature_names_after_selection": list(self.feature_names_after_selection),
            "reduction": dict(self.reduction_metadata),
            "final_selected_feature_indices": list(self.final_selected_feature_indices),
            "feature_names_out": list(self.feature_names_out),
        }
