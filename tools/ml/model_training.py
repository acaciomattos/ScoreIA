from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn

from tools.ml.interpretability import _permutation_importance
from tools.ml.lightgbm_compat import prepare_lightgbm_predict_input
from tools.ml.model_factory import ModelFactory
from tools.ml.runtime_config import configure_runtime_environment


configure_runtime_environment()


def _is_bucket_ensemble(model: Any) -> bool:
    return isinstance(model, dict) and model.get("kind") == "bucket_ensemble"


def get_primary_model(model: Any) -> Any:
    if _is_bucket_ensemble(model):
        return model.get("global_model")
    return model


def get_effective_model_kind(model: Any, fallback_model_kind: str) -> str:
    if _is_bucket_ensemble(model):
        return str(model.get("base_model_kind") or fallback_model_kind)
    return fallback_model_kind


def preparar_balanceamento_classificacao(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "none",
    random_state: int = 42,
    positive_class_weight: Optional[float] = None,
) -> Dict[str, Any]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    strategy = str(strategy or "none").lower()

    result = {
        "X": X_arr,
        "y": y_arr,
        "sample_weight": None,
        "strategy": strategy,
        "positive_class_weight": positive_class_weight,
    }

    unique, counts = np.unique(y_arr, return_counts=True)
    if unique.size != 2:
        return result

    class_counts = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
    classes_sorted = sorted(class_counts.keys())
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    n_minority = class_counts[minority_class]
    n_majority = class_counts[majority_class]

    if strategy == "class_weight":
        weights = np.ones(y_arr.shape[0], dtype=float)
        if positive_class_weight is not None:
            weights[y_arr == 1] = float(positive_class_weight)
        else:
            total = float(y_arr.shape[0])
            for cls in classes_sorted:
                cls_count = max(class_counts[cls], 1)
                weights[y_arr == cls] = total / (len(classes_sorted) * cls_count)
        result["sample_weight"] = weights
        return result

    rng = np.random.default_rng(random_state)
    idx_min = np.where(y_arr == minority_class)[0]
    idx_maj = np.where(y_arr == majority_class)[0]

    if strategy == "oversample" and n_minority > 0:
        extra = rng.choice(idx_min, size=max(0, n_majority - n_minority), replace=True)
        idx_final = np.concatenate([idx_maj, idx_min, extra])
        rng.shuffle(idx_final)
        result["X"] = X_arr[idx_final]
        result["y"] = y_arr[idx_final]
        return result

    if strategy == "undersample" and n_minority > 0:
        keep_majority = rng.choice(idx_maj, size=n_minority, replace=False)
        idx_final = np.concatenate([keep_majority, idx_min])
        rng.shuffle(idx_final)
        result["X"] = X_arr[idx_final]
        result["y"] = y_arr[idx_final]
        return result

    return result


def fit_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    params: Optional[Dict[str, Any]] = None,
    eval_set: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    early_stopping_rounds: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Any:
    model = ModelFactory.criar_lightgbm(task=task, params=params)
    metric_key = "rmse" if task == "regression" else "auc"

    fit_kwargs: Dict[str, Any] = {
        "eval_metric": metric_key,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    if eval_set:
        fit_kwargs["eval_set"] = list(eval_set)
        if early_stopping_rounds:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=max(1, int(early_stopping_rounds)), verbose=False),
                lgb.log_evaluation(period=0),
            ]

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_single_model(
    modelo_final: str,
    task: str,
    best_params: Optional[Dict[str, Any]],
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_oot: np.ndarray,
    early_stopping_rounds: int = 20,
    classification_balance_strategy: str = "none",
    positive_class_weight: Optional[float] = None,
) -> Dict[str, Any]:
    learning_curve = {
        "model": modelo_final,
        "metric_name": None,
        "values": [],
    }

    if task == "classification":
        balance = preparar_balanceamento_classificacao(
            X_train,
            y_train,
            strategy=classification_balance_strategy,
            random_state=42,
            positive_class_weight=positive_class_weight,
        )
        X_fit = balance["X"]
        y_fit = balance["y"]
        sample_weight = balance["sample_weight"]
    else:
        X_fit = X_train
        y_fit = y_train
        sample_weight = None

    predicoes: Dict[str, Any] = {
        "y_prob_train": None,
        "y_prob_test": None,
        "y_prob_oot": None,
        "y_pred_train": None,
        "y_pred_test": None,
        "y_pred_oot": None,
    }

    if modelo_final == "lightgbm":
        model = fit_lightgbm_model(
            X_train=X_fit,
            y_train=y_fit,
            task=task,
            params=best_params,
            eval_set=[(X_fit, y_fit)],
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=sample_weight,
        )

        evals_result = getattr(model, "evals_result_", {}) or {}
        if evals_result:
            dataset_key = next(iter(evals_result.keys()))
            metric_dict = evals_result.get(dataset_key, {}) or {}
            if metric_dict:
                curve_metric = next(iter(metric_dict.keys()))
                learning_curve = {
                    "model": modelo_final,
                    "metric_name": curve_metric,
                    "values": [float(v) for v in metric_dict.get(curve_metric, [])],
                }

        predicoes = predict_model_outputs(
            model=model,
            X_train=X_train,
            X_test=X_test,
            X_oot=X_oot,
            task=task,
            modelo_final=modelo_final,
        )
        return {
            "model": model,
            "learning_curve": learning_curve,
            "predicoes": predicoes,
        }

    if modelo_final != "mlp":
        raise ValueError("Modelo final invalido.")

    model = ModelFactory.criar_mlp(
        input_dim=input_dim,
        output_dim=1,
        params=best_params,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_t = torch.tensor(X_fit, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_fit, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=(best_params or {}).get("lr", 1e-3),
    )

    if task == "regression":
        criterion = torch.nn.MSELoss()
        sample_weight_tensor = None
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        if sample_weight is not None:
            sample_weight_tensor = torch.tensor(sample_weight, dtype=torch.float32).to(device)
        else:
            sample_weight_tensor = None

    train_loss_history = []
    n_epochs = int((best_params or {}).get("n_epochs", 200))

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(X_t).squeeze()
        loss = criterion(outputs, y_t)
        if sample_weight_tensor is not None:
            loss = (loss * sample_weight_tensor).mean()
        elif task == "classification":
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        train_loss_history.append(float(loss.item()))

    learning_curve = {
        "model": modelo_final,
        "metric_name": "loss",
        "values": train_loss_history,
    }

    predicoes = predict_model_outputs(
        model=model,
        X_train=X_train,
        X_test=X_test,
        X_oot=X_oot,
        task=task,
        modelo_final=modelo_final,
    )
    return {
        "model": model,
        "learning_curve": learning_curve,
        "predicoes": predicoes,
    }


def predict_single_matrix(
    model: Any,
    X: np.ndarray,
    task: str,
    modelo_final: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if modelo_final == "lightgbm":
        X_input = prepare_lightgbm_predict_input(model, X)
        if task == "classification":
            return None, model.predict_proba(X_input)[:, 1]
        return model.predict(X_input), None

    device = next(model.parameters()).device
    with torch.no_grad():
        raw = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().squeeze()

    if task == "classification":
        return None, 1.0 / (1.0 + np.exp(-raw))
    return raw, None


def predict_model_outputs(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_oot: np.ndarray,
    task: str,
    modelo_final: str,
    bucket_keys_train: Optional[Sequence[str]] = None,
    bucket_keys_test: Optional[Sequence[str]] = None,
    bucket_keys_oot: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not _is_bucket_ensemble(model):
        y_pred_train, y_prob_train = predict_single_matrix(model, X_train, task, modelo_final)
        y_pred_test, y_prob_test = predict_single_matrix(model, X_test, task, modelo_final)
        y_pred_oot, y_prob_oot = (None, None)
        if X_oot.shape[0] > 0:
            y_pred_oot, y_prob_oot = predict_single_matrix(model, X_oot, task, modelo_final)
        return {
            "y_prob_train": y_prob_train,
            "y_prob_test": y_prob_test,
            "y_prob_oot": y_prob_oot,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "y_pred_oot": y_pred_oot,
        }

    if modelo_final != "lightgbm":
        raise ValueError("Bucket ensemble suportado apenas para LightGBM.")

    global_model = model["global_model"]
    bucket_models = model.get("bucket_models", {})

    def _predict_bucketed(X: np.ndarray, bucket_keys: Optional[Sequence[str]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if task == "classification":
            y_prob = np.zeros(X.shape[0], dtype=float)
            if bucket_keys is None:
                y_prob[:] = global_model.predict_proba(
                    prepare_lightgbm_predict_input(global_model, X)
                )[:, 1]
                return None, y_prob
            bucket_keys_arr = np.asarray(bucket_keys, dtype=object)
            for bucket in np.unique(bucket_keys_arr):
                idx = np.where(bucket_keys_arr == bucket)[0]
                bucket_model = bucket_models.get(str(bucket), global_model)
                y_prob[idx] = bucket_model.predict_proba(
                    prepare_lightgbm_predict_input(bucket_model, X[idx])
                )[:, 1]
            return None, y_prob

        y_pred = np.zeros(X.shape[0], dtype=float)
        if bucket_keys is None:
            y_pred[:] = global_model.predict(
                prepare_lightgbm_predict_input(global_model, X)
            )
            return y_pred, None
        bucket_keys_arr = np.asarray(bucket_keys, dtype=object)
        for bucket in np.unique(bucket_keys_arr):
            idx = np.where(bucket_keys_arr == bucket)[0]
            bucket_model = bucket_models.get(str(bucket), global_model)
            y_pred[idx] = bucket_model.predict(
                prepare_lightgbm_predict_input(bucket_model, X[idx])
            )
        return y_pred, None

    y_pred_train, y_prob_train = _predict_bucketed(X_train, bucket_keys_train)
    y_pred_test, y_prob_test = _predict_bucketed(X_test, bucket_keys_test)
    y_pred_oot, y_prob_oot = (None, None)
    if X_oot.shape[0] > 0:
        y_pred_oot, y_prob_oot = _predict_bucketed(X_oot, bucket_keys_oot)

    return {
        "y_prob_train": y_prob_train,
        "y_prob_test": y_prob_test,
        "y_prob_oot": y_prob_oot,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_pred_oot": y_pred_oot,
    }


def train_bucket_ensemble(
    task: str,
    best_params: Optional[Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    bucket_keys_train: Sequence[str],
    early_stopping_rounds: int = 20,
    classification_balance_strategy: str = "none",
    positive_class_weight: Optional[float] = None,
    min_bucket_size: int = 30,
) -> Dict[str, Any]:
    if task != "classification" and task != "regression":
        raise ValueError("Task invalida.")

    if task == "classification":
        balance = preparar_balanceamento_classificacao(
            X_train,
            y_train,
            strategy=classification_balance_strategy,
            random_state=42,
            positive_class_weight=positive_class_weight,
        )
        X_fit = balance["X"]
        y_fit = balance["y"]
        sample_weight = balance["sample_weight"]
    else:
        X_fit = X_train
        y_fit = y_train
        sample_weight = None

    global_model = fit_lightgbm_model(
        X_train=X_fit,
        y_train=y_fit,
        task=task,
        params=best_params,
        eval_set=[(X_fit, y_fit)],
        early_stopping_rounds=early_stopping_rounds,
        sample_weight=sample_weight,
    )

    bucket_models: Dict[str, Any] = {}
    bucket_summary: Dict[str, Any] = {}
    bucket_keys_arr = np.asarray(bucket_keys_train, dtype=object)

    for bucket in np.unique(bucket_keys_arr):
        idx = np.where(bucket_keys_arr == bucket)[0]
        if idx.size < int(min_bucket_size):
            bucket_summary[str(bucket)] = {"status": "fallback_global", "n_samples": int(idx.size)}
            continue

        y_bucket = y_train[idx]
        if task == "classification" and np.unique(y_bucket).size < 2:
            bucket_summary[str(bucket)] = {"status": "fallback_global", "n_samples": int(idx.size)}
            continue

        X_bucket = X_train[idx]
        if task == "classification":
            bucket_balance = preparar_balanceamento_classificacao(
                X_bucket,
                y_bucket,
                strategy=classification_balance_strategy,
                random_state=42,
                positive_class_weight=positive_class_weight,
            )
            X_bucket_fit = bucket_balance["X"]
            y_bucket_fit = bucket_balance["y"]
            bucket_weight = bucket_balance["sample_weight"]
        else:
            X_bucket_fit = X_bucket
            y_bucket_fit = y_bucket
            bucket_weight = None

        bucket_model = fit_lightgbm_model(
            X_train=X_bucket_fit,
            y_train=y_bucket_fit,
            task=task,
            params=best_params,
            eval_set=[(X_bucket_fit, y_bucket_fit)],
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=bucket_weight,
        )
        bucket_models[str(bucket)] = bucket_model
        bucket_summary[str(bucket)] = {"status": "trained", "n_samples": int(idx.size)}

    evals_result = getattr(global_model, "evals_result_", {}) or {}
    learning_curve = {
        "model": "lightgbm_bucket_ensemble",
        "metric_name": None,
        "values": [],
    }
    if evals_result:
        dataset_key = next(iter(evals_result.keys()))
        metric_dict = evals_result.get(dataset_key, {}) or {}
        if metric_dict:
            curve_metric = next(iter(metric_dict.keys()))
            learning_curve = {
                "model": "lightgbm_bucket_ensemble",
                "metric_name": curve_metric,
                "values": [float(v) for v in metric_dict.get(curve_metric, [])],
            }

    return {
        "kind": "bucket_ensemble",
        "base_model_kind": "lightgbm",
        "global_model": global_model,
        "bucket_models": bucket_models,
        "bucket_summary": bucket_summary,
        "learning_curve": learning_curve,
    }


def rank_trained_model_features(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    modelo_final: str,
    random_state: int = 42,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    primary_model = get_primary_model(model)
    effective_kind = get_effective_model_kind(model, modelo_final)

    if effective_kind == "lightgbm" and hasattr(primary_model, "feature_importances_"):
        importancias = np.asarray(primary_model.feature_importances_, dtype=float)
    else:
        importancias = _permutation_importance(
            primary_model,
            np.asarray(X_train),
            np.asarray(y_train),
            task,
            effective_kind,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    if np.allclose(importancias, 0.0):
        importancias = np.nan_to_num(np.var(X_train, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    ranked_idx = np.argsort(-np.asarray(importancias, dtype=float))
    return {
        "ranked_indices": [int(idx) for idx in ranked_idx.tolist()],
        "importances": [float(v) for v in np.asarray(importancias, dtype=float).tolist()],
    }


def summarize_feature_importance_profile(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    modelo_final: str,
    feature_names: Optional[Sequence[str]] = None,
    cumulative_importance_cutoff_pct: float = 80.0,
    random_state: int = 42,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    ranking = rank_trained_model_features(
        model=model,
        X_train=X_train,
        y_train=y_train,
        task=task,
        modelo_final=modelo_final,
        random_state=random_state,
        n_repeats=n_repeats,
    )

    importances_raw = np.asarray(ranking["importances"], dtype=float)
    importance_mass = np.maximum(importances_raw, 0.0)
    if np.allclose(importance_mass, 0.0):
        importance_mass = np.abs(importances_raw)
    if np.allclose(importance_mass, 0.0):
        importance_mass = np.ones_like(importances_raw, dtype=float)

    ranked_indices = [int(idx) for idx in ranking["ranked_indices"]]
    active_ranked_indices = [idx for idx in ranked_indices if importance_mass[idx] > 0.0]
    if not active_ranked_indices:
        active_ranked_indices = list(ranked_indices)

    cutoff_pct = float(cumulative_importance_cutoff_pct or 80.0)
    cutoff_pct = min(max(cutoff_pct, 0.0), 100.0)

    active_scores = np.asarray([importance_mass[idx] for idx in active_ranked_indices], dtype=float)
    active_total = float(active_scores.sum()) or 1.0
    active_share = active_scores / active_total
    cumulative_share = np.cumsum(active_share)

    if active_ranked_indices:
        n_keep = int(np.searchsorted(cumulative_share, cutoff_pct / 100.0, side="left") + 1)
        n_keep = max(1, min(n_keep, len(active_ranked_indices)))
    else:
        n_keep = 0

    selected_ranked_indices = active_ranked_indices[:n_keep]
    tail_ranked_indices = active_ranked_indices[n_keep:]

    names = list(feature_names or [])

    def _name(idx: int) -> str:
        if 0 <= idx < len(names):
            return str(names[idx])
        return f"f_{idx}"

    importance_rows = []
    for order, idx in enumerate(active_ranked_indices, start=1):
        importance_rows.append(
            {
                "rank": order,
                "feature_index": int(idx),
                "feature_name": _name(idx),
                "importance_raw": float(importances_raw[idx]),
                "importance_mass": float(importance_mass[idx]),
                "importance_share": float(active_share[order - 1]),
                "importance_cumulative_share": float(cumulative_share[order - 1]),
                "selected_by_cutoff": bool(idx in selected_ranked_indices),
            }
        )

    return {
        "ranking": ranking,
        "active_feature_indices": [int(idx) for idx in active_ranked_indices],
        "active_feature_names": [_name(idx) for idx in active_ranked_indices],
        "selected_feature_indices": [int(idx) for idx in selected_ranked_indices],
        "selected_feature_names": [_name(idx) for idx in selected_ranked_indices],
        "tail_feature_indices": [int(idx) for idx in tail_ranked_indices],
        "tail_feature_names": [_name(idx) for idx in tail_ranked_indices],
        "n_input_features": int(importances_raw.size),
        "n_active_features": int(len(active_ranked_indices)),
        "n_selected_features": int(len(selected_ranked_indices)),
        "cutoff_pct": cutoff_pct,
        "importance_rows": importance_rows,
    }
