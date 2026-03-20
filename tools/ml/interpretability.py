import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from tools.ml.lightgbm_compat import prepare_lightgbm_predict_input
from tools.ml.metrics import (
    metricas_classificacao,
    metricas_regressao,
    otimizar_threshold_precision_recall,
)


def _feature_names(feature_names: Optional[List[str]], n_features: int) -> List[str]:
    if feature_names and len(feature_names) == n_features:
        return list(feature_names)
    return [f"feature_{i}" for i in range(n_features)]


def _predict_outputs(model, X: np.ndarray, task: str, modelo: str):
    if modelo == "lightgbm":
        X_input = prepare_lightgbm_predict_input(model, X)
        if task == "classification":
            y_prob = model.predict_proba(X_input)[:, 1]
            return None, y_prob
        y_pred = model.predict(X_input)
        return y_pred, None

    if modelo == "mlp":
        import torch

        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            raw = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().squeeze()

        if task == "classification":
            y_prob = 1.0 / (1.0 + np.exp(-raw))
            return None, y_prob

        return raw, None

    raise ValueError("Modelo nao suportado para interpretabilidade.")


def _metric_for_permutation(model, X: np.ndarray, y: np.ndarray, task: str, modelo: str) -> float:
    y_pred, y_prob = _predict_outputs(model, X, task, modelo)

    if task == "regression":
        return metricas_regressao(y, y_pred)["rmse"]

    return roc_auc_score(y, y_prob)


def _permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    modelo: str,
    n_repeats: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    X_base = np.asarray(X)

    base_score = _metric_for_permutation(model, X_base, y, task, modelo)
    importancias = np.zeros(X_base.shape[1], dtype=float)

    for j in range(X_base.shape[1]):
        perdas = []

        for _ in range(n_repeats):
            X_perm = X_base.copy()
            idx = rng.permutation(X_perm.shape[0])
            X_perm[:, j] = X_perm[idx, j]

            perm_score = _metric_for_permutation(model, X_perm, y, task, modelo)

            if task == "regression":
                perda = perm_score - base_score
            else:
                perda = base_score - perm_score

            perdas.append(perda)

        importancias[j] = float(np.mean(perdas))

    return importancias


def _top_features(feature_names: List[str], importancias: np.ndarray, top_n: int) -> List[Dict]:
    idx = np.argsort(importancias)[::-1][:top_n]
    return [
        {
            "feature": feature_names[i],
            "importance": float(importancias[i]),
        }
        for i in idx
    ]


def _predict_score_for_ale(model, X: np.ndarray, task: str, modelo: str) -> np.ndarray:
    y_pred, y_prob = _predict_outputs(model, X, task, modelo)
    return y_prob if task == "classification" else y_pred


def _ale_1d(
    model,
    X: np.ndarray,
    task: str,
    modelo: str,
    feature_idx: int,
    feature_name: str,
    n_bins: int = 10,
) -> Dict:
    x_col = X[:, feature_idx]
    quantis = np.unique(np.quantile(x_col, np.linspace(0, 1, n_bins + 1)))

    if quantis.size < 2:
        return {
            "feature": feature_name,
            "direction": "flat",
            "grid": [],
            "ale": [],
        }

    incrementos = np.zeros(quantis.size - 1, dtype=float)

    for k in range(quantis.size - 1):
        low = quantis[k]
        high = quantis[k + 1]

        if k == quantis.size - 2:
            mask = (x_col >= low) & (x_col <= high)
        else:
            mask = (x_col >= low) & (x_col < high)

        if not np.any(mask):
            continue

        X_low = X[mask].copy()
        X_high = X[mask].copy()

        X_low[:, feature_idx] = low
        X_high[:, feature_idx] = high

        pred_low = _predict_score_for_ale(model, X_low, task, modelo)
        pred_high = _predict_score_for_ale(model, X_high, task, modelo)

        incrementos[k] = float(np.mean(pred_high - pred_low))

    ale = np.cumsum(incrementos)
    ale = ale - np.mean(ale)

    delta = float(ale[-1] - ale[0]) if ale.size > 1 else 0.0
    if delta > 0:
        direction = "positive"
    elif delta < 0:
        direction = "negative"
    else:
        direction = "flat"

    return {
        "feature": feature_name,
        "direction": direction,
        "grid": quantis[1:].astype(float).tolist(),
        "ale": ale.astype(float).tolist(),
    }


def _shap_like_summary_from_matrix(
    shap_matrix: np.ndarray,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_names: List[str],
    top_n: int,
    random_state: int,
    scatter_points_per_feature: int,
    method: str,
) -> Dict:
    rng = np.random.default_rng(random_state)

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1][:top_n]

    top_shap = []
    scatter = []

    for i in idx_sorted:
        x_col = X_sample[:, i]
        s_col = shap_matrix[:, i]

        if np.std(x_col) > 0 and np.std(s_col) > 0:
            corr_feat_shap = float(np.corrcoef(x_col, s_col)[0, 1])
        else:
            corr_feat_shap = 0.0

        if np.std(x_col) > 0 and np.std(y_sample) > 0:
            corr_feat_target = float(np.corrcoef(x_col, y_sample)[0, 1])
        else:
            corr_feat_target = 0.0

        if corr_feat_shap > 0:
            direction = "positive"
        elif corr_feat_shap < 0:
            direction = "negative"
        else:
            direction = "flat"

        inversao = bool(corr_feat_shap * corr_feat_target < 0)

        top_shap.append(
            {
                "feature": feature_names[i],
                "mean_abs_shap": float(mean_abs[i]),
                "effect_direction": direction,
                "corr_feature_shap": corr_feat_shap,
                "corr_feature_target": corr_feat_target,
                "effect_inversion_vs_target": inversao,
            }
        )

        if x_col.size > scatter_points_per_feature:
            idx_pts = rng.choice(x_col.size, size=scatter_points_per_feature, replace=False)
            x_scatter = x_col[idx_pts]
            s_scatter = s_col[idx_pts]
        else:
            x_scatter = x_col
            s_scatter = s_col

        scatter.append(
            {
                "feature": feature_names[i],
                "x": x_scatter.astype(float).tolist(),
                "shap": s_scatter.astype(float).tolist(),
            }
        )

    return {
        "available": True,
        "method": method,
        "sample_size": int(X_sample.shape[0]),
        "top_shap": top_shap,
        "scatter": scatter,
    }


def _tree_shap_summary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_n: int,
    sample_size: int,
    random_state: int,
    scatter_points_per_feature: int,
) -> Dict:
    rng = np.random.default_rng(random_state)

    if X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]
    else:
        X_sample = X
        y_sample = y

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
                category=UserWarning,
            )
            shap_values = explainer.shap_values(
                prepare_lightgbm_predict_input(model, X_sample)
            )

        if isinstance(shap_values, list):
            shap_matrix = np.asarray(shap_values[-1])
        else:
            shap_matrix = np.asarray(shap_values)

        if shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[:, :, -1]

        return _shap_like_summary_from_matrix(
            shap_matrix=shap_matrix,
            X_sample=X_sample,
            y_sample=y_sample,
            feature_names=feature_names,
            top_n=top_n,
            random_state=random_state,
            scatter_points_per_feature=scatter_points_per_feature,
            method="shap_tree_explainer",
        )

    except Exception as exc:
        # Fallback sem dependencia do pacote shap para LightGBM
        try:
            contrib = model.predict(
                prepare_lightgbm_predict_input(model, X_sample),
                pred_contrib=True,
            )
            shap_matrix = np.asarray(contrib)

            if shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, :, -1]

            # LightGBM retorna uma coluna extra de bias no final
            if shap_matrix.shape[1] == len(feature_names) + 1:
                shap_matrix = shap_matrix[:, :-1]

            if shap_matrix.shape[1] != len(feature_names):
                raise ValueError("pred_contrib_shape_mismatch")

            out = _shap_like_summary_from_matrix(
                shap_matrix=shap_matrix,
                X_sample=X_sample,
                y_sample=y_sample,
                feature_names=feature_names,
                top_n=top_n,
                random_state=random_state,
                scatter_points_per_feature=scatter_points_per_feature,
                method="lightgbm_pred_contrib_fallback",
            )
            out["fallback_reason"] = f"shap_unavailable:{exc.__class__.__name__}"
            return out

        except Exception as fallback_exc:
            return {
                "available": False,
                "reason": f"shap_unavailable:{exc.__class__.__name__};fallback_failed:{fallback_exc.__class__.__name__}",
                "top_shap": [],
                "scatter": [],
            }


def gerar_interpretabilidade(
    model,
    X_train,
    y_train,
    task: str,
    modelo: str,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    n_repeats: int = 5,
    ale_top_k: int = 20,
    shap_sample_size: int = 1000,
    shap_scatter_points: int = 200,
    random_state: int = 42,
    threshold_classificacao: Optional[float] = None,
) -> Dict:
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    names = _feature_names(feature_names, X.shape[1])

    resultado = {
        "task": task,
        "modelo": modelo,
        "top_n": int(top_n),
    }

    threshold_usado = None

    if task == "classification":
        _, y_prob_train = _predict_outputs(model, X, task, modelo)

        if threshold_classificacao is None:
            threshold_usado = float(otimizar_threshold_precision_recall(y, y_prob_train))
        else:
            threshold_usado = float(threshold_classificacao)

        y_hat = (y_prob_train >= threshold_usado).astype(int)
        resultado["threshold_classificacao_usado"] = threshold_usado
        resultado["metricas_treino_threshold_usado"] = metricas_classificacao(
            y,
            y_hat,
            y_prob=y_prob_train,
        )

    if modelo == "lightgbm" and hasattr(model, "feature_importances_"):
        importancias = np.asarray(model.feature_importances_, dtype=float)
        resultado["importancia_global"] = {
            "metodo": "lightgbm_feature_importance",
            "top_features": _top_features(names, importancias, top_n),
        }

        resultado["shap"] = _tree_shap_summary(
            model,
            X,
            y,
            names,
            top_n=top_n,
            sample_size=shap_sample_size,
            random_state=random_state,
            scatter_points_per_feature=shap_scatter_points,
        )

    else:
        importancias_perm = _permutation_importance(
            model,
            X,
            y,
            task,
            modelo,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        top_global = _top_features(names, importancias_perm, top_n)

        resultado["importancia_global"] = {
            "metodo": "permutation_importance",
            "n_repeats": int(n_repeats),
            "top_features": top_global,
        }

        top_ale = []
        for item in top_global[:ale_top_k]:
            feat = item["feature"]
            idx = names.index(feat)
            top_ale.append(
                _ale_1d(
                    model,
                    X,
                    task,
                    modelo,
                    feature_idx=idx,
                    feature_name=feat,
                )
            )

        resultado["ale"] = {
            "available": True,
            "top_features_ale": top_ale,
        }

        resultado["shap"] = {
            "available": False,
            "reason": "not_default_for_mlp",
            "top_shap": [],
            "scatter": [],
        }

    return resultado


def salvar_interpretabilidade_json(
    interpretabilidade: Dict,
    output_dir: str = "reports",
    file_name: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"interpretability_{timestamp}.json"

    path = os.path.join(output_dir, file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(interpretabilidade, f, ensure_ascii=False, indent=2)

    return path
