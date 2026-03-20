import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score

from tools.ml.lightgbm_compat import prepare_lightgbm_predict_input
from tools.ml.model_factory import ModelFactory
from tools.ml.runtime_config import configure_runtime_environment


configure_runtime_environment()


def _get_metric_key(task: str) -> str:
    return "rmse" if task == "regression" else "auc"


def _direction(task: str) -> str:
    return "minimize" if task == "regression" else "maximize"


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _normalize_importance_cutoff(value: float) -> float:
    cutoff = float(value)
    if cutoff > 1.0:
        cutoff = cutoff / 100.0
    return float(min(max(cutoff, 0.01), 1.0))


def build_temporal_forward_folds(
    temporal_month_keys: Sequence[Any],
    min_train_months: int = 2,
    max_folds: int = 6,
    min_train_size: int = 20,
    min_val_size: int = 10,
) -> Dict[str, Any]:
    month_keys = [None if v is None else str(v) for v in temporal_month_keys]
    months_sorted = sorted({m for m in month_keys if m is not None})

    fold_defs: List[Dict[str, Any]] = []

    if len(months_sorted) <= min_train_months:
        return {
            "valid": False,
            "reason": (
                f"insufficient_months: n_months={len(months_sorted)}, "
                f"min_train_months={min_train_months}"
            ),
            "months_sorted": months_sorted,
            "fold_defs": fold_defs,
        }

    start_pos = max(1, int(min_train_months))
    for pos in range(start_pos, len(months_sorted)):
        val_month = months_sorted[pos]
        train_months = months_sorted[:pos]

        train_idx = np.asarray(
            [i for i, m in enumerate(month_keys) if m in train_months],
            dtype=int,
        )
        val_idx = np.asarray(
            [i for i, m in enumerate(month_keys) if m == val_month],
            dtype=int,
        )

        if train_idx.size < min_train_size or val_idx.size < min_val_size:
            continue

        fold_defs.append(
            {
                "fold_id": len(fold_defs) + 1,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_months": train_months,
                "val_month": val_month,
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
            }
        )

    if max_folds > 0 and len(fold_defs) > max_folds:
        fold_defs = fold_defs[-max_folds:]
        for i, fd in enumerate(fold_defs):
            fd["fold_id"] = i + 1

    if not fold_defs:
        return {
            "valid": False,
            "reason": "no_valid_temporal_folds",
            "months_sorted": months_sorted,
            "fold_defs": [],
        }

    return {
        "valid": True,
        "reason": None,
        "months_sorted": months_sorted,
        "fold_defs": fold_defs,
    }


def _fit_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    early_stopping_rounds: int,
    random_state: int,
) -> Any:
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": int(random_state),
        "n_jobs": -1,
        "verbosity": -1,
    }
    metric_key = _get_metric_key(task)

    model = ModelFactory.criar_lightgbm(task=task, params=params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=max(1, int(early_stopping_rounds)), verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=metric_key,
        callbacks=callbacks,
    )
    return model


def _score_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
) -> Optional[float]:
    X_input = prepare_lightgbm_predict_input(model, X)
    if task == "classification":
        if np.unique(y).size < 2:
            return None
        y_prob = model.predict_proba(X_input)[:, 1]
        return _safe_float(roc_auc_score(y, y_prob))

    y_pred = model.predict(X_input)
    return _safe_float(np.sqrt(mean_squared_error(y, y_pred)))


def _select_features_by_cumulative_importance(
    importances: np.ndarray,
    cutoff: float,
) -> Tuple[List[int], List[Dict[str, float]]]:
    values = np.asarray(importances, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if values.size == 0:
        return [], []

    total = float(np.sum(values))
    if total <= 0.0:
        shares = np.repeat(1.0 / values.size, values.size)
    else:
        shares = values / total

    ranked_idx = np.argsort(-shares)
    selected: List[int] = []
    diagnostics: List[Dict[str, float]] = []
    cumulative = 0.0

    for idx in ranked_idx:
        share = float(shares[idx])
        cumulative += share
        selected.append(int(idx))
        diagnostics.append(
            {
                "feature_index": int(idx),
                "importance": float(values[idx]),
                "importance_share": share,
                "importance_cumulative_share": float(cumulative),
            }
        )
        if cumulative >= cutoff:
            break

    if not selected and ranked_idx.size > 0:
        idx = int(ranked_idx[0])
        selected = [idx]
        diagnostics = [
            {
                "feature_index": idx,
                "importance": float(values[idx]),
                "importance_share": float(shares[idx]),
                "importance_cumulative_share": float(shares[idx]),
            }
        ]

    return selected, diagnostics


def _is_train_val_gap_acceptable(
    train_score: float,
    val_score: float,
    direction: str,
    rel_tolerance: float,
    abs_tolerance: float,
) -> Tuple[bool, float, float]:
    tol = max(abs_tolerance, abs(train_score) * rel_tolerance)
    if direction == "maximize":
        gap = max(0.0, float(train_score) - float(val_score))
    else:
        gap = max(0.0, float(val_score) - float(train_score))
    return gap <= tol, float(gap), float(tol)


def _build_stable_importance_core(
    X_train: np.ndarray,
    y_train: np.ndarray,
    folds: Sequence[Dict[str, Any]],
    task: str,
    early_stopping_rounds: int,
    random_state: int,
    performance_rel_tolerance: float,
    performance_abs_tolerance: float,
    importance_cumulative_cutoff: float,
) -> Dict[str, Any]:
    direction = _direction(task)
    accepted_feature_sets: List[set] = []
    fold_diagnostics: List[Dict[str, Any]] = []

    for fold_pos, fd in enumerate(folds, start=1):
        train_idx = fd["train_idx"]
        val_idx = fd["val_idx"]

        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        model = _fit_lightgbm_model(
            X_train=X_fold_train,
            y_train=y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            task=task,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state + fold_pos,
        )

        train_score = _score_model(model, X_fold_train, y_fold_train, task)
        val_score = _score_model(model, X_fold_val, y_fold_val, task)

        feature_importances = np.asarray(
            getattr(model, "feature_importances_", np.zeros(X_train.shape[1])),
            dtype=float,
        )
        selected_idx, selected_diag = _select_features_by_cumulative_importance(
            feature_importances,
            cutoff=importance_cumulative_cutoff,
        )

        accepted = False
        performance_gap = None
        performance_gap_tolerance = None

        if train_score is not None and val_score is not None:
            accepted, performance_gap, performance_gap_tolerance = _is_train_val_gap_acceptable(
                train_score=train_score,
                val_score=val_score,
                direction=direction,
                rel_tolerance=performance_rel_tolerance,
                abs_tolerance=performance_abs_tolerance,
            )

        if accepted and selected_idx:
            accepted_feature_sets.append(set(selected_idx))

        fold_diagnostics.append(
            {
                "fold_id": int(fd["fold_id"]),
                "train_months": list(fd["train_months"]),
                "val_month": fd["val_month"],
                "n_train": int(fd["n_train"]),
                "n_val": int(fd["n_val"]),
                "train_score": train_score,
                "val_score": val_score,
                "performance_gap": performance_gap,
                "performance_gap_tolerance": performance_gap_tolerance,
                "accepted_for_intersection": bool(accepted),
                "selected_feature_indices_by_importance": selected_idx,
                "selected_feature_importance_details": selected_diag,
            }
        )

    intersection_idx: List[int] = []
    if accepted_feature_sets:
        intersection_idx = sorted(set.intersection(*accepted_feature_sets))

    return {
        "accepted_folds": int(len(accepted_feature_sets)),
        "fold_diagnostics": fold_diagnostics,
        "stable_intersection_feature_indices": intersection_idx,
    }


def _fit_lightgbm_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    early_stopping_rounds: int,
    random_state: int,
) -> Optional[float]:
    model = _fit_lightgbm_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task=task,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )
    return _score_model(model, X_val, y_val, task)


def _evaluate_candidate(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: np.ndarray,
    folds: Sequence[Dict[str, Any]],
    task: str,
    early_stopping_rounds: int,
    random_state: int,
) -> Dict[str, Any]:
    fold_scores: List[float] = []
    used_folds = 0

    for fd in folds:
        train_idx = fd["train_idx"]
        val_idx = fd["val_idx"]

        X_train = X[train_idx][:, feature_idx]
        X_val = X[val_idx][:, feature_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        score = _fit_lightgbm_and_score(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task=task,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )

        if score is None:
            continue

        fold_scores.append(float(score))
        used_folds += 1

    if not fold_scores:
        return {
            "scores": [],
            "n_folds_valid": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "cv": None,
        }

    arr = np.asarray(fold_scores, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    vrange = float(vmax - vmin)
    cv = float(std / (abs(mean) + 1e-12))

    return {
        "scores": fold_scores,
        "n_folds_valid": int(used_folds),
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "range": vrange,
        "cv": cv,
    }


def _build_candidate_sizes(
    n_features: int,
    max_candidates: int = 8,
    max_features: Optional[int] = None,
) -> List[int]:
    if n_features <= 0:
        return []

    upper = n_features if max_features is None else max(1, min(int(max_features), n_features))

    seeds = [10, 20, 40, 60, 80, 120, 180, upper]
    if upper != n_features:
        seeds.append(n_features)

    sizes = sorted({s for s in seeds if 1 <= s <= n_features})

    if len(sizes) > max_candidates:
        idx = np.linspace(0, len(sizes) - 1, num=max_candidates)
        sizes = sorted({sizes[int(round(i))] for i in idx})

    return sizes


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

    imp = np.asarray(getattr(model, "feature_importances_", np.zeros(X_train.shape[1])), dtype=float)
    if imp.size != X_train.shape[1] or np.allclose(imp, 0.0):
        imp = np.nan_to_num(np.var(X_train, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    return np.argsort(-imp)


def _is_close_to_best(
    mean_score: float,
    best_score: float,
    direction: str,
    rel_tolerance: float,
    abs_tolerance: float,
) -> Tuple[bool, float]:
    tol = max(abs_tolerance, abs(best_score) * rel_tolerance)
    if direction == "maximize":
        gap = best_score - mean_score
        return (gap <= tol), float(gap)
    gap = mean_score - best_score
    return (gap <= tol), float(gap)


def selecionar_features_por_estabilidade_temporal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    temporal_month_keys: Sequence[Any],
    feature_names: Sequence[str],
    task: str,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
    max_folds: int = 6,
    max_candidates: int = 8,
    max_features: Optional[int] = None,
    min_train_months: int = 2,
    min_train_size: int = 20,
    min_val_size: int = 10,
    min_valid_folds_per_candidate: int = 2,
    performance_rel_tolerance: float = 0.05,
    performance_abs_tolerance: Optional[float] = None,
    importance_cumulative_cutoff: float = 80.0,
) -> Dict[str, Any]:
    n_samples, n_features = X_train.shape

    if n_features == 0:
        return {
            "enabled": True,
            "applied": False,
            "reason": "no_features",
            "n_features_original": 0,
            "n_features_selected": 0,
            "selected_feature_indices": [],
            "selected_feature_names": [],
            "candidate_results": [],
        }

    if len(feature_names) != n_features:
        feature_names = [f"f_{i}" for i in range(n_features)]

    folds_info = build_temporal_forward_folds(
        temporal_month_keys=temporal_month_keys,
        min_train_months=min_train_months,
        max_folds=max_folds,
        min_train_size=min_train_size,
        min_val_size=min_val_size,
    )

    if not folds_info["valid"]:
        return {
            "enabled": True,
            "applied": False,
            "reason": folds_info["reason"],
            "n_features_original": int(n_features),
            "n_features_selected": int(n_features),
            "selected_feature_indices": list(range(n_features)),
            "selected_feature_names": list(feature_names),
            "candidate_results": [],
            "metric_key": _get_metric_key(task),
            "direction": _direction(task),
            "temporal_cv": {
                "n_samples": int(n_samples),
                "n_months": len(folds_info.get("months_sorted", [])),
                "months": folds_info.get("months_sorted", []),
                "n_folds": 0,
                "folds": [],
            },
        }

    ranked_idx = _rank_features_lightgbm(
        X_train=X_train,
        y_train=y_train,
        task=task,
        random_state=random_state,
    )

    metric_key = _get_metric_key(task)
    direction = _direction(task)

    if performance_abs_tolerance is None:
        performance_abs_tolerance = 0.002 if task == "classification" else 1e-4

    importance_cumulative_cutoff = _normalize_importance_cutoff(importance_cumulative_cutoff)

    stable_importance_core = _build_stable_importance_core(
        X_train=X_train,
        y_train=y_train,
        folds=folds_info["fold_defs"],
        task=task,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        performance_rel_tolerance=performance_rel_tolerance,
        performance_abs_tolerance=performance_abs_tolerance,
        importance_cumulative_cutoff=importance_cumulative_cutoff,
    )

    stable_core_idx = stable_importance_core.get("stable_intersection_feature_indices", [])
    stable_core_fallback_used = False
    candidate_base_idx = ranked_idx

    if stable_core_idx:
        stable_core_set = {int(idx) for idx in stable_core_idx}
        candidate_base_idx = np.asarray(
            [idx for idx in ranked_idx if int(idx) in stable_core_set],
            dtype=int,
        )
    else:
        stable_core_fallback_used = True

    candidate_n_features = int(candidate_base_idx.size)
    candidate_sizes = _build_candidate_sizes(
        n_features=candidate_n_features,
        max_candidates=max_candidates,
        max_features=max_features,
    )
    if candidate_n_features not in candidate_sizes:
        candidate_sizes.append(candidate_n_features)
        candidate_sizes = sorted(set(candidate_sizes))

    candidate_results: List[Dict[str, Any]] = []

    for top_k in candidate_sizes:
        idx = candidate_base_idx[:top_k]
        stats = _evaluate_candidate(
            X=X_train,
            y=y_train,
            feature_idx=idx,
            folds=folds_info["fold_defs"],
            task=task,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )

        candidate_results.append(
            {
                "top_k": int(top_k),
                "n_features": int(top_k),
                "feature_indices": idx.astype(int).tolist(),
                "feature_names": [str(feature_names[i]) for i in idx],
                **stats,
            }
        )

    valid_candidates = [
        c
        for c in candidate_results
        if c.get("mean") is not None and int(c.get("n_folds_valid", 0)) >= int(min_valid_folds_per_candidate)
    ]

    if not valid_candidates:
        return {
            "enabled": True,
            "applied": False,
            "reason": "no_candidate_with_min_valid_folds",
            "n_features_original": int(n_features),
            "n_features_selected": int(n_features),
            "selected_feature_indices": list(range(n_features)),
            "selected_feature_names": list(feature_names),
            "metric_key": metric_key,
            "direction": direction,
            "importance_cumulative_cutoff": float(importance_cumulative_cutoff),
            "stable_importance_core": stable_importance_core,
            "stable_core_fallback_used": bool(stable_core_fallback_used),
            "candidate_results": candidate_results,
            "temporal_cv": {
                "n_samples": int(n_samples),
                "n_months": len(folds_info.get("months_sorted", [])),
                "months": folds_info.get("months_sorted", []),
                "n_folds": len(folds_info.get("fold_defs", [])),
                "folds": [
                    {
                        "fold_id": fd["fold_id"],
                        "train_months": fd["train_months"],
                        "val_month": fd["val_month"],
                        "n_train": fd["n_train"],
                        "n_val": fd["n_val"],
                    }
                    for fd in folds_info.get("fold_defs", [])
                ],
            },
        }

    if direction == "maximize":
        best_score = max(float(c["mean"]) for c in valid_candidates)
    else:
        best_score = min(float(c["mean"]) for c in valid_candidates)

    eligible: List[Dict[str, Any]] = []
    for c in valid_candidates:
        is_close, gap = _is_close_to_best(
            mean_score=float(c["mean"]),
            best_score=float(best_score),
            direction=direction,
            rel_tolerance=float(performance_rel_tolerance),
            abs_tolerance=float(performance_abs_tolerance),
        )
        c["performance_gap_to_best"] = float(gap)
        c["performance_close_to_best"] = bool(is_close)
        if is_close:
            eligible.append(c)

    pool = eligible if eligible else valid_candidates
    selected = min(
        pool,
        key=lambda c: (
            float(c.get("std") if c.get("std") is not None else 1e9),
            float(c.get("cv") if c.get("cv") is not None else 1e9),
            int(c.get("n_features", n_features)),
        ),
    )

    selected_idx = [int(i) for i in selected["feature_indices"]]
    applied = len(selected_idx) < n_features
    reason = "selected_by_temporal_stability"
    if not applied:
        selected_idx = list(range(n_features))
        reason = "no_reduction_after_selection"

    selected_idx_set = set(selected_idx)

    for c in candidate_results:
        c["selected"] = set(c["feature_indices"]) == selected_idx_set

    return {
        "enabled": True,
        "applied": bool(applied),
        "reason": reason,
        "selection_rule": (
            "Choose candidates near best mean score (within tolerance), "
            "then minimize dispersion (std, cv), then prefer fewer features."
        ),
        "selector_model": "lightgbm",
        "metric_key": metric_key,
        "direction": direction,
        "importance_cumulative_cutoff": float(importance_cumulative_cutoff),
        "best_mean_score": float(best_score),
        "performance_rel_tolerance": float(performance_rel_tolerance),
        "performance_abs_tolerance": float(performance_abs_tolerance),
        "n_features_original": int(n_features),
        "n_features_selected": len(selected_idx),
        "selected_feature_indices": selected_idx,
        "selected_feature_names": [str(feature_names[i]) for i in selected_idx],
        "stable_importance_core": stable_importance_core,
        "stable_core_fallback_used": bool(stable_core_fallback_used),
        "candidate_results": candidate_results,
        "temporal_cv": {
            "n_samples": int(n_samples),
            "n_months": len(folds_info.get("months_sorted", [])),
            "months": folds_info.get("months_sorted", []),
            "n_folds": len(folds_info.get("fold_defs", [])),
            "folds": [
                {
                    "fold_id": fd["fold_id"],
                    "train_months": fd["train_months"],
                    "val_month": fd["val_month"],
                    "n_train": fd["n_train"],
                    "n_val": fd["n_val"],
                }
                for fd in folds_info.get("fold_defs", [])
            ],
        },
    }



