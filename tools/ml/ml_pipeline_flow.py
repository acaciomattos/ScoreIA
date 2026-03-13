import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import KFold, StratifiedKFold

from tools.ml.artifact_export import exportar_artefatos_execucao
from tools.ml.artifacts import (
    DEFAULT_ML_ARTIFACTS_DIR,
    criar_run_context,
    garantir_diretorios_artefatos,
    nome_artefato,
    resolver_caminhos_artefatos,
    resolver_caminhos_artefatos_estagio,
)
from tools.ml.data_pipeline import DataPipeline
from tools.ml.feature_selection import selecionar_features_por_estabilidade_temporal
from tools.ml.feature_postprocessing import FeaturePostProcessor
from tools.ml.interpretability import gerar_interpretabilidade, salvar_interpretabilidade_json
from tools.ml.metrics import (
    metricas_classificacao,
    metricas_regressao,
    otimizar_threshold_precision_recall,
)
from tools.ml.model_factory import ModelFactory
from tools.ml.model_training import (
    get_effective_model_kind,
    get_primary_model,
    predict_model_outputs,
    summarize_feature_importance_profile,
    train_bucket_ensemble,
    train_single_model,
)
from tools.ml.optimization import otimizar_modelo_cv
from tools.ml.reporting import gerar_relatorio_tecnico_html, salvar_relatorio_html


def _np_list(values: Sequence[Any], cast: Optional[type] = None) -> List[Any]:
    arr = np.asarray(values).reshape(-1)
    if cast is int:
        return [int(v) for v in arr.tolist()]
    if cast is float:
        return [float(v) for v in arr.tolist()]
    return arr.tolist()


def _build_scored_subset(
    df_raw: Optional[pl.DataFrame],
    split_group: str,
    task: str,
    y_pred_vals: Sequence[Any],
    y_prob_vals: Optional[Sequence[Any]] = None,
    temporal_month_col: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    if df_raw is None or df_raw.height == 0:
        return None

    frame = df_raw.with_columns(pl.lit(split_group).alias("split_group"))

    if temporal_month_col and temporal_month_col in frame.columns:
        frame = frame.with_columns(
            pl.col(temporal_month_col).cast(pl.Utf8).alias("temporal_month")
        )

    if task == "classification":
        frame = frame.with_columns(
            [
                pl.Series("predicted_probability", _np_list(y_prob_vals, float)),
                pl.Series("predicted_class", _np_list(y_pred_vals, int)),
            ]
        )
    else:
        frame = frame.with_columns(
            [
                pl.Series("predicted_value", _np_list(y_pred_vals, float)),
            ]
        )

    return frame


def _segment_key(record: Dict[str, Any], segments: Sequence[str]) -> Optional[str]:
    if not segments:
        return None
    parts = []
    for seg in segments:
        parts.append(f"{seg}={record.get(seg)}")
    return " | ".join(parts)


def _build_distribution(
    records: Sequence[Dict[str, Any]],
    target_col: str,
    axis: str,
    segment_features: Sequence[str],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], Dict[str, float]] = {}

    for row in records:
        split_group = row.get("split_group")
        month = row.get("temporal_month")
        seg_key = _segment_key(row, segment_features)

        if axis == "tempo":
            if month is None:
                continue
            key = (split_group, str(month))
        elif axis == "segmento":
            if seg_key is None:
                continue
            key = (split_group, seg_key)
        elif axis == "tempo_segmento":
            if month is None or seg_key is None:
                continue
            key = (split_group, str(month), seg_key)
        else:
            continue

        bucket = buckets.setdefault(key, {"n": 0.0, "target_sum": 0.0})
        bucket["n"] += 1.0
        bucket["target_sum"] += float(row.get(target_col) or 0.0)

    out: List[Dict[str, Any]] = []
    for key, agg in buckets.items():
        if axis == "tempo":
            split_group, month = key
            out.append(
                {
                    "split_group": split_group,
                    "temporal_month": month,
                    "volume": int(agg["n"]),
                    "target_mean": (agg["target_sum"] / agg["n"]) if agg["n"] else None,
                }
            )
        elif axis == "segmento":
            split_group, seg_key = key
            out.append(
                {
                    "split_group": split_group,
                    "segment_key": seg_key,
                    "volume": int(agg["n"]),
                    "target_mean": (agg["target_sum"] / agg["n"]) if agg["n"] else None,
                }
            )
        else:
            split_group, month, seg_key = key
            out.append(
                {
                    "split_group": split_group,
                    "temporal_month": month,
                    "segment_key": seg_key,
                    "volume": int(agg["n"]),
                    "target_mean": (agg["target_sum"] / agg["n"]) if agg["n"] else None,
                }
            )

    sort_cols = ["split_group", "temporal_month", "segment_key"]
    out.sort(key=lambda row: tuple(str(row.get(col, "")) for col in sort_cols))
    return out


def _build_metrics_by_axis(
    records: Sequence[Dict[str, Any]],
    task: str,
    target_col: str,
    axis: str,
    segment_features: Sequence[str],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], Dict[str, List[Any]]] = {}

    for row in records:
        split_group = row.get("split_group")
        month = row.get("temporal_month")
        seg_key = _segment_key(row, segment_features)

        if axis == "tempo":
            if month is None:
                continue
            key = (split_group, str(month))
        elif axis == "segmento":
            if seg_key is None:
                continue
            key = (split_group, seg_key)
        elif axis == "tempo_segmento":
            if month is None or seg_key is None:
                continue
            key = (split_group, str(month), seg_key)
        else:
            continue

        bucket = buckets.setdefault(key, {"y_true": [], "y_pred": [], "y_prob": []})
        bucket["y_true"].append(row.get(target_col))

        if task == "classification":
            bucket["y_pred"].append(row.get("predicted_class"))
            bucket["y_prob"].append(row.get("predicted_probability"))
        else:
            bucket["y_pred"].append(row.get("predicted_value"))

    out: List[Dict[str, Any]] = []
    for key, bucket in buckets.items():
        if len(bucket["y_true"]) < 2:
            continue

        if task == "classification":
            metricas_local = metricas_classificacao(
                np.asarray(bucket["y_true"]),
                np.asarray(bucket["y_pred"]),
                y_prob=np.asarray(bucket["y_prob"]),
            )
        else:
            metricas_local = metricas_regressao(
                np.asarray(bucket["y_true"]),
                np.asarray(bucket["y_pred"]),
            )

        if axis == "tempo":
            split_group, month = key
            row_out = {"split_group": split_group, "temporal_month": month, "n_obs": len(bucket["y_true"])}
        elif axis == "segmento":
            split_group, seg_key = key
            row_out = {"split_group": split_group, "segment_key": seg_key, "n_obs": len(bucket["y_true"])}
        else:
            split_group, month, seg_key = key
            row_out = {
                "split_group": split_group,
                "temporal_month": month,
                "segment_key": seg_key,
                "n_obs": len(bucket["y_true"]),
            }

        row_out.update(metricas_local)
        out.append(row_out)

    sort_cols = ["split_group", "temporal_month", "segment_key"]
    out.sort(key=lambda row: tuple(str(row.get(col, "")) for col in sort_cols))
    return out


def _resumo_estabilidade(
    metric_rows: Dict[str, List[Dict[str, Any]]],
    metric_names: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    resumo: Dict[str, Dict[str, Dict[str, float]]] = {}

    for eixo, rows in metric_rows.items():
        eixo_resumo: Dict[str, Dict[str, float]] = {}
        for metric_name in metric_names:
            vals = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
            if not vals:
                continue
            eixo_resumo[metric_name] = {
                "n_buckets": float(len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
            }
        resumo[eixo] = eixo_resumo

    return resumo


def inicializar_execucao_ml(config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    run_context = criar_run_context(config.get("run_timestamp"))
    caminhos_artefatos = resolver_caminhos_artefatos(
        run_context=run_context,
        artifacts_dir=config.get("artifacts_dir", DEFAULT_ML_ARTIFACTS_DIR),
    )
    os.makedirs(caminhos_artefatos["run_dir"], exist_ok=True)
    return run_context, caminhos_artefatos


def carregar_pipeline_e_particionar(config: Dict[str, Any]) -> Dict[str, Any]:
    balance_strategy = str(config.get("classification_balance_strategy", "none"))
    if config.get("task") != "classification" and balance_strategy != "none":
        raise ValueError("As tecnicas de weight/under/oversampling estao disponiveis apenas para classificacao.")

    if str(config.get("segment_strategy", "global")) == "per_bucket_model" and not (config.get("segment_features") or []):
        raise ValueError("O modo 'um modelo por bucket' requer ao menos uma variavel de segmentacao.")

    if not bool(config.get("apply_missing_treatment", True)) and config.get("modelo") in {"mlp", "auto"}:
        raise ValueError(
            "Sem tratamento de missings, o pipeline suporta apenas LightGBM nesta etapa. "
            "Ative a imputacao ou selecione LightGBM."
        )

    pipeline = DataPipeline(config["caminho"])
    pipeline.carregar(config["formato"])

    temporal_feature = config.get("temporal_feature")
    segment_features = [s for s in (config.get("segment_features") or []) if s]
    oot_months = int(config.get("oot_months", 0) or 0)
    segment_strategy = config.get("segment_strategy", "global")

    X_train_raw, X_test_raw, y_train, y_test, X_oot_raw, y_oot, split_info = pipeline.split_holdout(
        target=config["target"],
        test_size=0.2,
        task=config["task"],
        temporal_feature=temporal_feature,
        oot_months=oot_months,
        segment_features=segment_features,
        segment_strategy=segment_strategy,
    )

    X_train = pipeline.fit_transform(
        X_train_raw,
        modelo=config["modelo"],
        aplicar_tratamento_missing=bool(config.get("apply_missing_treatment", True)),
    )
    X_test = pipeline.transform(X_test_raw)
    X_oot = pipeline.transform(X_oot_raw) if len(y_oot) else np.empty((0, X_train.shape[1]))

    return {
        "pipeline": pipeline,
        "split_info": split_info,
        "X_train": X_train,
        "X_test": X_test,
        "X_oot": X_oot,
        "X_train_base": X_train,
        "X_test_base": X_test,
        "X_oot_base": X_oot,
        "y_train": y_train,
        "y_test": y_test,
        "y_oot": y_oot,
    }


def aplicar_feature_selection_temporal_ml(
    config: Dict[str, Any],
    pipeline: DataPipeline,
    split_info: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_oot: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    feature_names_model = list(getattr(pipeline, "feature_names_out", []))
    feature_selection_stability = {
        "enabled": False,
        "applied": False,
        "reason": "disabled_or_no_temporal_feature",
        "n_features_original": int(X_train.shape[1]),
        "n_features_selected": int(X_train.shape[1]),
        "selected_feature_indices": list(range(X_train.shape[1])),
        "selected_feature_names": feature_names_model,
        "candidate_results": [],
    }

    temporal_feature = config.get("temporal_feature")
    selecionar_estavel = bool(config.get("stability_feature_selection", True)) and bool(temporal_feature)

    if selecionar_estavel:
        temporal_month_col = split_info.get("temporal_month_col")
        df_train_raw = split_info.get("df_train_raw")
        temporal_month_keys = (
            df_train_raw[temporal_month_col].to_list()
            if (df_train_raw is not None and temporal_month_col and temporal_month_col in df_train_raw.columns)
            else []
        )

        feature_selection_stability = selecionar_features_por_estabilidade_temporal(
            X_train=X_train,
            y_train=y_train,
            temporal_month_keys=temporal_month_keys,
            feature_names=feature_names_model,
            task=config["task"],
            early_stopping_rounds=int(config.get("early_stopping_rounds", 20)),
            random_state=int(config.get("seed", 42)),
            max_folds=int(config.get("stability_max_folds", 6)),
            max_candidates=int(config.get("stability_max_candidates", 8)),
            max_features=config.get("stability_max_features"),
            min_train_months=int(config.get("stability_min_train_months", 2)),
            min_train_size=int(config.get("stability_min_train_size", 20)),
            min_val_size=int(config.get("stability_min_val_size", 10)),
            min_valid_folds_per_candidate=int(config.get("stability_min_valid_folds", 2)),
            performance_rel_tolerance=float(config.get("stability_perf_rel_tolerance", 0.05)),
            performance_abs_tolerance=config.get("stability_perf_abs_tolerance"),
            importance_cumulative_cutoff=float(config.get("stability_importance_cutoff_pct", 80.0)),
        )
    else:
        feature_selection_stability["reason"] = (
            "temporal_feature_not_informed" if not temporal_feature else "disabled_by_config"
        )

    selected_feature_indices = feature_selection_stability.get("selected_feature_indices") or list(range(X_train.shape[1]))
    selected_feature_indices = [int(idx) for idx in selected_feature_indices]

    selected_feature_names = [feature_names_model[idx] for idx in selected_feature_indices] if feature_names_model else []

    pipeline.metadata["n_features_original"] = int(
        feature_selection_stability.get("n_features_original", X_train.shape[1])
    )
    pipeline.metadata["n_features_selected"] = int(
        feature_selection_stability.get("n_features_selected", X_train.shape[1])
    )
    pipeline.metadata["selected_feature_names"] = list(selected_feature_names)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_oot": X_oot,
        "feature_selection_stability": feature_selection_stability,
        "selected_feature_indices": selected_feature_indices,
        "selected_feature_names": selected_feature_names,
    }


def aplicar_pos_processamento_features_ml(
    config: Dict[str, Any],
    X_train_base: np.ndarray,
    X_test_base: np.ndarray,
    X_oot_base: np.ndarray,
    y_train: np.ndarray,
    feature_names_model: Sequence[str],
    selected_feature_indices: Sequence[int],
) -> Dict[str, Any]:
    postprocessor = FeaturePostProcessor()
    X_train = postprocessor.fit(
        X_train=X_train_base,
        y_train=y_train,
        feature_names=feature_names_model,
        selected_feature_indices=selected_feature_indices,
        task=config["task"],
        random_state=int(config.get("seed", 42)),
    )
    X_test = postprocessor.transform(X_test_base)
    X_oot = postprocessor.transform(X_oot_base) if X_oot_base.shape[0] > 0 else np.empty((0, X_train.shape[1]))

    return {
        "postprocessor": postprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "X_oot": X_oot,
        "feature_names_out": list(postprocessor.feature_names_out),
        "postprocessing_metadata": postprocessor.metadata(),
    }


def _transformar_matrizes_estagio(
    postprocessor: FeaturePostProcessor,
    X_train_base: np.ndarray,
    X_test_base: np.ndarray,
    X_oot_base: np.ndarray,
) -> Dict[str, np.ndarray]:
    X_train = postprocessor.transform(X_train_base)
    X_test = postprocessor.transform(X_test_base)
    X_oot = (
        postprocessor.transform(X_oot_base)
        if X_oot_base.shape[0] > 0
        else np.empty((0, X_train.shape[1]))
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_oot": X_oot,
    }


def derivar_postprocessor_pruning_ml(
    config: Dict[str, Any],
    source_stage: Dict[str, Any],
    X_train_base: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    if not bool(config.get("prune_best_model", False)):
        return {"executed": False, "reason": "disabled"}

    importance_profile = summarize_feature_importance_profile(
        model=source_stage["model"],
        X_train=source_stage["X_train"],
        y_train=y_train,
        task=config["task"],
        modelo_final=source_stage["effective_model_kind"],
        feature_names=source_stage["selected_feature_names"],
        cumulative_importance_cutoff_pct=float(config.get("prune_importance_cutoff_pct", 80.0)),
        random_state=int(config.get("seed", 42)),
        n_repeats=int(config.get("prune_importance_n_repeats", 3)),
    )

    selected_indices = importance_profile["selected_feature_indices"]
    source_feature_count = int(len(source_stage["selected_feature_names"]))

    if len(selected_indices) < 1 or len(selected_indices) >= source_feature_count:
        return {
            "executed": False,
            "reason": "invalid_or_no_effect",
            "importance_profile": importance_profile,
        }

    postprocessor = FeaturePostProcessor()
    postprocessor.fit(
        X_train=X_train_base,
        y_train=y_train,
        feature_names=source_stage["selected_feature_names"],
        selected_feature_indices=selected_indices,
        task=config["task"],
        random_state=int(config.get("seed", 42)),
        upstream=source_stage["postprocessor"],
    )

    return {
        "executed": True,
        "reason": "enabled",
        "postprocessor": postprocessor,
        "importance_profile": importance_profile,
        "summary": {
            "executed": True,
            "method": "cumulative_importance_cutoff",
            "source_stage": source_stage["stage_name"],
            "cutoff_pct": float(config.get("prune_importance_cutoff_pct", 80.0)),
            "n_source_features": source_feature_count,
            "n_selected_features": int(len(selected_indices)),
            "selected_feature_indices": [int(idx) for idx in selected_indices],
            "selected_feature_names": list(postprocessor.feature_names_out),
            "importance_profile": importance_profile,
        },
    }


def derivar_postprocessor_reducao_ml(
    config: Dict[str, Any],
    source_stage: Dict[str, Any],
    X_train_base: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    if not bool(config.get("reduce_tail_features", False)):
        return {"executed": False, "reason": "disabled"}

    importance_profile = summarize_feature_importance_profile(
        model=source_stage["model"],
        X_train=source_stage["X_train"],
        y_train=y_train,
        task=config["task"],
        modelo_final=source_stage["effective_model_kind"],
        feature_names=source_stage["selected_feature_names"],
        cumulative_importance_cutoff_pct=100.0,
        random_state=int(config.get("seed", 42)),
        n_repeats=int(config.get("prune_importance_n_repeats", 3)),
    )

    active_indices = importance_profile["active_feature_indices"]
    source_feature_count = int(len(active_indices))
    tail_count = int(config.get("reduction_tail_features_count", 0) or 0)

    if source_feature_count < 3 or tail_count < 2:
        return {
            "executed": False,
            "reason": "invalid_or_no_effect",
            "importance_profile": importance_profile,
        }

    tail_count = min(tail_count, source_feature_count - 1)
    if tail_count < 2:
        return {
            "executed": False,
            "reason": "invalid_or_no_effect",
            "importance_profile": importance_profile,
        }

    reduced_source_indices = active_indices
    tail_indices_relative = list(range(source_feature_count - tail_count, source_feature_count))
    head_count = source_feature_count - tail_count

    postprocessor = FeaturePostProcessor()
    postprocessor.fit(
        X_train=X_train_base,
        y_train=y_train,
        feature_names=source_stage["selected_feature_names"],
        selected_feature_indices=reduced_source_indices,
        task=config["task"],
        reduction_enabled=True,
        reduction_tail_indices=tail_indices_relative,
        reduction_variance_threshold=float(config.get("reduction_variance_threshold", 0.70)),
        random_state=int(config.get("seed", 42)),
        upstream=source_stage["postprocessor"],
    )

    reduction_meta = postprocessor.metadata().get("reduction", {})
    if not reduction_meta.get("enabled"):
        return {
            "executed": False,
            "reason": "invalid_or_no_effect",
            "importance_profile": importance_profile,
        }

    return {
        "executed": True,
        "reason": "enabled",
        "postprocessor": postprocessor,
        "importance_profile": importance_profile,
        "summary": {
            "executed": True,
            "method": "tail_pca",
            "source_stage": source_stage["stage_name"],
            "n_source_features": int(source_feature_count),
            "n_head_features_kept": int(head_count),
            "n_tail_features_reduced": int(tail_count),
            "n_components_generated": int(reduction_meta.get("n_components_applied", 0)),
            "variance_threshold": float(config.get("reduction_variance_threshold", 0.70)),
            "active_feature_indices": [int(idx) for idx in active_indices],
            "active_feature_names": [source_stage["selected_feature_names"][idx] for idx in active_indices],
            "tail_feature_indices": [int(idx) for idx in active_indices[-tail_count:]],
            "tail_feature_names": [source_stage["selected_feature_names"][idx] for idx in active_indices[-tail_count:]],
            "output_feature_names": list(postprocessor.feature_names_out),
            "importance_profile": importance_profile,
            "reduction_metadata": reduction_meta,
        },
    }


def montar_resumo_dados_ml(
    split_info: Dict[str, Any],
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_oot: np.ndarray,
    feature_selection_stability: Dict[str, Any],
    temporal_feature: Optional[str],
) -> Dict[str, Any]:
    n_train = int(len(y_train))
    n_test = int(len(y_test))
    n_oot = int(len(y_oot))
    n_total = n_train + n_test + n_oot
    temporal_validation = split_info.get("temporal_validation") or {}

    return {
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "n_oot": n_oot,
        "pct_train": float((n_train / n_total) * 100.0) if n_total else 0.0,
        "pct_test": float((n_test / n_total) * 100.0) if n_total else 0.0,
        "pct_oot": float((n_oot / n_total) * 100.0) if n_total else 0.0,
        "target_mean_train": float(np.mean(y_train)) if len(y_train) else None,
        "target_mean_test": float(np.mean(y_test)) if len(y_test) else None,
        "target_mean_oot": float(np.mean(y_oot)) if len(y_oot) else None,
        "temporal_feature": temporal_feature,
        "temporal_n_months": temporal_validation.get("n_months"),
        "temporal_months_oot": split_info.get("oot_month_list", []),
        "excluded_from_model": split_info.get("excluded_from_model", []),
        "stability_feature_selection_applied": bool(feature_selection_stability.get("applied")),
        "stability_feature_selection_reason": feature_selection_stability.get("reason"),
        "n_features_original": int(feature_selection_stability.get("n_features_original", 0)),
        "n_features_selected": int(feature_selection_stability.get("n_features_selected", 0)),
    }


def criar_folds_cv_ml(
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if task == "classification":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    folds = []
    for train_idx, val_idx in splitter.split(X_train, y_train):
        folds.append((X_train[train_idx], X_train[val_idx], y_train[train_idx], y_train[val_idx]))
    return folds


def executar_otimizacao_ml(
    config: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    resultado_otimizacao = None
    best_params = None
    best_cv_score = None
    modelo_final = config["modelo"]

    if config.get("otimizar", False):
        resultado_otimizacao = otimizar_modelo_cv(
            folds=folds,
            task=config["task"],
            modelo=config["modelo"],
            n_trials=config.get("n_trials", 30),
            early_stopping_rounds=int(config.get("early_stopping_rounds", 20)),
            classification_balance_strategy=config.get("classification_balance_strategy", "none"),
            positive_class_weight=config.get("positive_class_weight"),
        )

        if config["modelo"] == "auto":
            modelo_final = resultado_otimizacao["melhor_modelo"]
            best_params = resultado_otimizacao["resultado"]["best_params"]
            best_cv_score = resultado_otimizacao["resultado"]["best_score"]
        else:
            best_params = resultado_otimizacao["best_params"]
            best_cv_score = resultado_otimizacao["best_score"]

    return {
        "resultado_otimizacao": resultado_otimizacao,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "modelo_final": modelo_final,
    }


def treinar_modelo_final_ml(
    config: Dict[str, Any],
    modelo_final: str,
    best_params: Optional[Dict[str, Any]],
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_oot: np.ndarray,
    split_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    split_info = split_info or {}
    segment_strategy = str(config.get("segment_strategy", "global"))
    classification_balance_strategy = str(config.get("classification_balance_strategy", "none"))
    positive_class_weight = config.get("positive_class_weight")

    if segment_strategy == "per_bucket_model":
        if modelo_final != "lightgbm":
            raise ValueError("O modo 'um modelo por bucket' esta suportado apenas para LightGBM nesta etapa.")

        bucket_keys_train = split_info.get("segment_bucket_train", [])
        bucket_keys_test = split_info.get("segment_bucket_test", [])
        bucket_keys_oot = split_info.get("segment_bucket_oot", [])

        model = train_bucket_ensemble(
            task=config["task"],
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            bucket_keys_train=bucket_keys_train,
            early_stopping_rounds=int(config.get("early_stopping_rounds", 20)),
            classification_balance_strategy=classification_balance_strategy,
            positive_class_weight=positive_class_weight,
            min_bucket_size=int(config.get("segment_bucket_min_samples", 30)),
        )
        predicoes = predict_model_outputs(
            model=model,
            X_train=X_train,
            X_test=X_test,
            X_oot=X_oot,
            task=config["task"],
            modelo_final="lightgbm",
            bucket_keys_train=bucket_keys_train,
            bucket_keys_test=bucket_keys_test,
            bucket_keys_oot=bucket_keys_oot,
        )
        return {
            "model": model,
            "learning_curve": model.get("learning_curve", {}),
            "predicoes": predicoes,
            "effective_model_kind": "lightgbm",
        }

    treino = train_single_model(
        modelo_final=modelo_final,
        task=config["task"],
        best_params=best_params,
        input_dim=input_dim,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        X_oot=X_oot,
        early_stopping_rounds=int(config.get("early_stopping_rounds", 20)),
        classification_balance_strategy=classification_balance_strategy,
        positive_class_weight=positive_class_weight,
    )
    treino["effective_model_kind"] = modelo_final
    return treino


def calcular_metricas_finais_ml(
    config: Dict[str, Any],
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_oot: np.ndarray,
    predicoes: Dict[str, Any],
) -> Dict[str, Any]:
    threshold_otimo = None
    metricas_oot = None

    if config["task"] == "regression":
        metricas_treino = metricas_regressao(y_train, predicoes["y_pred_train"])
        metricas_teste = metricas_regressao(y_test, predicoes["y_pred_test"])
        if len(y_oot):
            metricas_oot = metricas_regressao(y_oot, predicoes["y_pred_oot"])
    else:
        threshold_otimo = config.get("threshold_classificacao")
        if threshold_otimo is None:
            threshold_otimo = otimizar_threshold_precision_recall(y_train, predicoes["y_prob_train"])

        predicoes["y_pred_train"] = (predicoes["y_prob_train"] >= threshold_otimo).astype(int)
        predicoes["y_pred_test"] = (predicoes["y_prob_test"] >= threshold_otimo).astype(int)
        if predicoes.get("y_prob_oot") is not None:
            predicoes["y_pred_oot"] = (predicoes["y_prob_oot"] >= threshold_otimo).astype(int)

        metricas_treino = metricas_classificacao(
            y_train,
            predicoes["y_pred_train"],
            y_prob=predicoes["y_prob_train"],
        )
        metricas_teste = metricas_classificacao(
            y_test,
            predicoes["y_pred_test"],
            y_prob=predicoes["y_prob_test"],
        )
        if len(y_oot):
            metricas_oot = metricas_classificacao(
                y_oot,
                predicoes["y_pred_oot"],
                y_prob=predicoes["y_prob_oot"],
            )

    return {
        "metricas_treino": metricas_treino,
        "metricas_teste": metricas_teste,
        "metricas_oot": metricas_oot,
        "threshold_otimo": threshold_otimo,
        "predicoes": predicoes,
    }


def gerar_dataset_scored_ml(
    config: Dict[str, Any],
    pipeline: DataPipeline,
    split_info: Dict[str, Any],
    model: Any,
    modelo_final: str,
    postprocessor: FeaturePostProcessor,
    threshold_otimo: Optional[float],
    run_context: Dict[str, str],
    caminhos_artefatos: Dict[str, str],
) -> str:
    df_scoring = pipeline.df
    if getattr(pipeline, "segment_bucket_col", None) in (pipeline.feature_order or []) and getattr(pipeline, "segment_bucket_col", None) not in df_scoring.columns:
        bucket_values = DataPipeline.build_segment_bucket_values(
            df_scoring,
            split_info.get("segment_features", []),
        )
        df_scoring = df_scoring.with_columns(pl.Series(pipeline.segment_bucket_col, bucket_values))

    X_full_raw = df_scoring.select(pipeline.feature_order).to_numpy()
    X_full = pipeline.transform(X_full_raw)
    X_full = postprocessor.transform(X_full)

    bucket_keys_full = None
    if split_info.get("segment_strategy") == "per_bucket_model":
        bucket_keys_full = DataPipeline.build_segment_bucket_values(
            pipeline.df,
            split_info.get("segment_features", []),
        )

    predicoes_full = predict_model_outputs(
        model=model,
        X_train=X_full,
        X_test=X_full,
        X_oot=np.empty((0, X_full.shape[1])),
        task=config["task"],
        modelo_final=get_effective_model_kind(model, modelo_final),
        bucket_keys_train=bucket_keys_full,
        bucket_keys_test=bucket_keys_full,
    )

    if config["task"] == "classification":
        y_prob_full = predicoes_full["y_prob_train"]
        y_pred_full = (y_prob_full >= threshold_otimo).astype(int)
        scored_df = df_scoring.with_columns(
            [
                pl.Series("predicted_probability", np.asarray(y_prob_full, dtype=float).tolist()),
                pl.Series("predicted_class", np.asarray(y_pred_full, dtype=int).tolist()),
            ]
        )
    else:
        y_pred_full = predicoes_full["y_pred_train"]
        scored_df = df_scoring.with_columns(
            [
                pl.Series("predicted_value", np.asarray(y_pred_full, dtype=float).tolist()),
            ]
        )

    path = os.path.join(
        caminhos_artefatos["run_scored_dir"],
        nome_artefato("scored_dataset", run_context, "csv"),
    )
    scored_df.write_csv(path)
    return path


def gerar_analise_tempo_segmento_ml(
    config: Dict[str, Any],
    split_info: Dict[str, Any],
    predicoes: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    temporal_feature = config.get("temporal_feature")
    temporal_month_col = split_info.get("temporal_month_col")
    df_train_raw = split_info.get("df_train_raw")
    df_test_raw = split_info.get("df_test_raw")
    df_oot_raw = split_info.get("df_oot_raw")
    segment_features_valid = split_info.get("segment_features", [])

    subset_frames = []
    train_frame = _build_scored_subset(
        df_train_raw,
        "train",
        config["task"],
        predicoes["y_pred_train"],
        predicoes.get("y_prob_train"),
        temporal_month_col=temporal_month_col,
    )
    if train_frame is not None:
        subset_frames.append(train_frame)

    test_frame = _build_scored_subset(
        df_test_raw,
        "test",
        config["task"],
        predicoes["y_pred_test"],
        predicoes.get("y_prob_test"),
        temporal_month_col=temporal_month_col,
    )
    if test_frame is not None:
        subset_frames.append(test_frame)

    if predicoes.get("y_pred_oot") is not None:
        oot_frame = _build_scored_subset(
            df_oot_raw,
            "oot",
            config["task"],
            predicoes["y_pred_oot"],
            predicoes.get("y_prob_oot"),
            temporal_month_col=temporal_month_col,
        )
        if oot_frame is not None:
            subset_frames.append(oot_frame)

    if not subset_frames or not (temporal_feature or segment_features_valid):
        return None

    df_subsets = pl.concat(subset_frames, how="vertical_relaxed")

    base_cols = ["split_group", config["target"]]
    if temporal_feature and "temporal_month" in df_subsets.columns:
        base_cols.append("temporal_month")
    for seg in segment_features_valid:
        if seg in df_subsets.columns:
            base_cols.append(seg)

    if config["task"] == "classification":
        base_cols.extend(["predicted_class", "predicted_probability"])
    else:
        base_cols.append("predicted_value")

    records = df_subsets.select(base_cols).to_dicts()

    dist_tempo = _build_distribution(records, config["target"], "tempo", segment_features_valid)
    dist_segmento = _build_distribution(records, config["target"], "segmento", segment_features_valid)
    dist_tempo_segmento = _build_distribution(records, config["target"], "tempo_segmento", segment_features_valid)

    met_tempo = _build_metrics_by_axis(records, config["task"], config["target"], "tempo", segment_features_valid)
    met_segmento = _build_metrics_by_axis(records, config["task"], config["target"], "segmento", segment_features_valid)
    met_tempo_segmento = _build_metrics_by_axis(
        records,
        config["task"],
        config["target"],
        "tempo_segmento",
        segment_features_valid,
    )

    metric_names = ["auc", "f1"] if config["task"] == "classification" else ["rmse", "r2"]
    estabilidade = _resumo_estabilidade(
        {
            "tempo": met_tempo,
            "segmento": met_segmento,
            "tempo_segmento": met_tempo_segmento,
        },
        metric_names,
    )

    return {
        "distribuicao_tempo": dist_tempo,
        "distribuicao_segmento": dist_segmento,
        "distribuicao_tempo_segmento": dist_tempo_segmento,
        "metricas_por_tempo": met_tempo,
        "metricas_por_segmento": met_segmento,
        "metricas_por_tempo_segmento": met_tempo_segmento,
        "estabilidade_resumo": estabilidade,
    }


def gerar_interpretabilidade_ml(
    config: Dict[str, Any],
    caminhos_artefatos: Dict[str, str],
    run_context: Dict[str, str],
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    modelo_final: str,
    selected_feature_names: Sequence[str],
    threshold_otimo: Optional[float],
) -> Dict[str, Any]:
    effective_model_kind = get_effective_model_kind(model, modelo_final)
    primary_model = get_primary_model(model)
    interpretabilidade = gerar_interpretabilidade(
        model=primary_model,
        X_train=X_train,
        y_train=y_train,
        task=config["task"],
        modelo=effective_model_kind,
        feature_names=selected_feature_names,
        top_n=config.get("interpretability_top_n", 20),
        n_repeats=config.get("interpretability_n_repeats", 5),
        ale_top_k=config.get("interpretability_ale_top_k", 20),
        shap_sample_size=config.get("interpretability_shap_sample_size", 1000),
        random_state=42,
        threshold_classificacao=threshold_otimo,
    )

    path = salvar_interpretabilidade_json(
        interpretabilidade,
        output_dir=caminhos_artefatos["run_reports_dir"],
        file_name=nome_artefato("interpretability", run_context, "json"),
    )
    return {
        "interpretabilidade": interpretabilidade,
        "caminho_interpretabilidade": path,
    }


def gerar_relatorio_ml(
    run_context: Dict[str, str],
    caminhos_artefatos: Dict[str, str],
    config: Dict[str, Any],
    resumo_dados: Dict[str, Any],
    metricas_treino: Dict[str, Any],
    metricas_teste: Dict[str, Any],
    metricas_oot: Optional[Dict[str, Any]],
    modelo_final: str,
    best_params: Optional[Dict[str, Any]],
    best_cv_score: Optional[float],
    threshold_otimo: Optional[float],
    metadata: Dict[str, Any],
    interpretabilidade: Dict[str, Any],
    resultado_otimizacao: Optional[Dict[str, Any]],
    learning_curve: Dict[str, Any],
    analise_tempo_segmento: Optional[Dict[str, Any]],
    feature_selection_stability: Dict[str, Any],
    file_suffix: Optional[str] = None,
) -> str:
    relatorio_html = gerar_relatorio_tecnico_html(
        run_context=run_context,
        config=config,
        resumo_dados=resumo_dados,
        metricas_treino=metricas_treino,
        metricas_teste=metricas_teste,
        metricas_oot=metricas_oot,
        modelo_final=modelo_final,
        best_params=best_params,
        best_cv_score=best_cv_score,
        threshold_classificacao=threshold_otimo,
        metadata=metadata,
        interpretabilidade=interpretabilidade,
        resultado_otimizacao=resultado_otimizacao,
        learning_curve=learning_curve,
        analise_tempo_segmento=analise_tempo_segmento,
        feature_selection_stability=feature_selection_stability,
    )

    file_name = None
    if file_suffix:
        file_name = f"report_{run_context['timestamp']}{file_suffix}.html"

    return salvar_relatorio_html(
        html_content=relatorio_html,
        output_dir=caminhos_artefatos["run_reports_dir"],
        run_context=run_context,
        file_name=file_name,
    )


def executar_estagio_modelagem_ml(
    stage_name: str,
    config: Dict[str, Any],
    pipeline: DataPipeline,
    split_info: Dict[str, Any],
    X_train_base: np.ndarray,
    X_test_base: np.ndarray,
    X_oot_base: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_oot: np.ndarray,
    postprocessor: FeaturePostProcessor,
    run_context: Dict[str, str],
    caminhos_artefatos_root: Dict[str, str],
    resumo_dados_base: Dict[str, Any],
    feature_selection_stability: Dict[str, Any],
    source_stage_name: Optional[str] = None,
    stage_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    caminhos_estagio = resolver_caminhos_artefatos_estagio(
        caminhos_artefatos_root,
        stage_name,
    )
    garantir_diretorios_artefatos(caminhos_estagio)

    matrizes = _transformar_matrizes_estagio(
        postprocessor=postprocessor,
        X_train_base=X_train_base,
        X_test_base=X_test_base,
        X_oot_base=X_oot_base,
    )
    X_train = matrizes["X_train"]
    X_test = matrizes["X_test"]
    X_oot = matrizes["X_oot"]
    selected_feature_names = list(postprocessor.feature_names_out)

    folds = criar_folds_cv_ml(config["task"], X_train, y_train)
    resultado_otimizacao_pack = executar_otimizacao_ml(config, folds)
    best_params = resultado_otimizacao_pack["best_params"]
    best_cv_score = resultado_otimizacao_pack["best_cv_score"]
    modelo_final = resultado_otimizacao_pack["modelo_final"]
    resultado_otimizacao = resultado_otimizacao_pack["resultado_otimizacao"]

    treino_final = treinar_modelo_final_ml(
        config=config,
        modelo_final=modelo_final,
        best_params=best_params,
        input_dim=X_train.shape[1],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        X_oot=X_oot,
        split_info=split_info,
    )
    model = treino_final["model"]
    learning_curve = treino_final["learning_curve"]
    predicoes = treino_final["predicoes"]
    effective_model_kind = treino_final.get("effective_model_kind", modelo_final)

    resultado_metricas = calcular_metricas_finais_ml(
        config=config,
        y_train=y_train,
        y_test=y_test,
        y_oot=y_oot,
        predicoes=predicoes,
    )
    metricas_treino = resultado_metricas["metricas_treino"]
    metricas_teste = resultado_metricas["metricas_teste"]
    metricas_oot = resultado_metricas["metricas_oot"]
    threshold_otimo = resultado_metricas["threshold_otimo"]
    predicoes = resultado_metricas["predicoes"]

    analise_tempo_segmento = gerar_analise_tempo_segmento_ml(
        config=config,
        split_info=split_info,
        predicoes=predicoes,
    )

    resultado_interpretabilidade = gerar_interpretabilidade_ml(
        config=config,
        caminhos_artefatos=caminhos_estagio,
        run_context=run_context,
        model=model,
        X_train=X_train,
        y_train=y_train,
        modelo_final=effective_model_kind,
        selected_feature_names=selected_feature_names,
        threshold_otimo=threshold_otimo,
    )
    interpretabilidade = resultado_interpretabilidade["interpretabilidade"]
    caminho_interpretabilidade = resultado_interpretabilidade["caminho_interpretabilidade"]

    metadata_estagio = deepcopy(getattr(pipeline, "metadata", {}))
    metadata_estagio["feature_postprocessing"] = postprocessor.metadata()
    metadata_estagio["model_stage"] = {
        "stage_name": stage_name,
        "source_stage_name": source_stage_name,
        "summary": deepcopy(stage_summary or {}),
    }
    if isinstance(model, dict) and model.get("kind") == "bucket_ensemble":
        metadata_estagio["bucket_summary"] = model.get("bucket_summary", {})

    resumo_dados = deepcopy(resumo_dados_base)
    resumo_dados["model_stage_name"] = stage_name
    resumo_dados["source_stage_name"] = source_stage_name
    resumo_dados["n_features_model_input"] = int(len(selected_feature_names))

    config_relatorio = dict(config)
    config_relatorio["model_stage_name"] = stage_name
    config_relatorio["source_stage_name"] = source_stage_name

    caminho_relatorio_html = gerar_relatorio_ml(
        run_context=run_context,
        caminhos_artefatos=caminhos_estagio,
        config=config_relatorio,
        resumo_dados=resumo_dados,
        metricas_treino=metricas_treino,
        metricas_teste=metricas_teste,
        metricas_oot=metricas_oot,
        modelo_final=effective_model_kind,
        best_params=best_params,
        best_cv_score=best_cv_score,
        threshold_otimo=threshold_otimo,
        metadata=metadata_estagio,
        interpretabilidade=interpretabilidade,
        resultado_otimizacao=resultado_otimizacao,
        learning_curve=learning_curve,
        analise_tempo_segmento=analise_tempo_segmento,
        feature_selection_stability=feature_selection_stability,
    )

    caminho_dataset_scored = gerar_dataset_scored_ml(
        config=config,
        pipeline=pipeline,
        split_info=split_info,
        model=model,
        modelo_final=effective_model_kind,
        postprocessor=postprocessor,
        threshold_otimo=threshold_otimo,
        run_context=run_context,
        caminhos_artefatos=caminhos_estagio,
    )

    artefatos_exportados = exportar_artefatos_ml(
        model=model,
        pipeline=pipeline,
        task=config["task"],
        modelo_final=effective_model_kind,
        run_context=run_context,
        caminhos_artefatos=caminhos_estagio,
        config=config_relatorio,
        metricas_treino=metricas_treino,
        metricas_teste=metricas_teste,
        threshold_classificacao=threshold_otimo,
        best_params=best_params,
        best_cv_score=best_cv_score,
        resultado_otimizacao=resultado_otimizacao,
        input_dim=len(selected_feature_names),
        interpretabilidade=interpretabilidade,
        caminho_interpretabilidade_run=caminho_interpretabilidade,
        caminho_relatorio_html_run=caminho_relatorio_html,
        caminho_scored_dataset_run=caminho_dataset_scored,
        postprocessor=postprocessor,
        pipeline_metadata=metadata_estagio,
    )

    return {
        "stage_name": stage_name,
        "source_stage_name": source_stage_name,
        "summary": deepcopy(stage_summary or {}),
        "model": model,
        "learning_curve": learning_curve,
        "predicoes": predicoes,
        "metricas_treino": metricas_treino,
        "metricas": metricas_teste,
        "metricas_oot": metricas_oot,
        "threshold_classificacao": threshold_otimo,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "modelo_final": effective_model_kind,
        "resultado_otimizacao": resultado_otimizacao,
        "postprocessor": postprocessor,
        "postprocessor_metadata": postprocessor.metadata(),
        "selected_feature_names": selected_feature_names,
        "analise_tempo_segmento": analise_tempo_segmento,
        "interpretabilidade": interpretabilidade,
        "metadata": metadata_estagio,
        "resumo_dados": resumo_dados,
        "caminho_interpretabilidade": caminho_interpretabilidade,
        "caminho_relatorio_html": caminho_relatorio_html,
        "caminho_dataset_scored": caminho_dataset_scored,
        "artefatos_exportados": artefatos_exportados,
        "caminhos_artefatos": caminhos_estagio,
        "effective_model_kind": effective_model_kind,
        "X_train": X_train,
        "X_test": X_test,
        "X_oot": X_oot,
    }


def exportar_artefatos_ml(
    model: Any,
    pipeline: DataPipeline,
    task: str,
    modelo_final: str,
    run_context: Dict[str, str],
    caminhos_artefatos: Dict[str, str],
    config: Dict[str, Any],
    metricas_treino: Dict[str, Any],
    metricas_teste: Dict[str, Any],
    threshold_classificacao: Optional[float],
    best_params: Optional[Dict[str, Any]],
    best_cv_score: Optional[float],
    resultado_otimizacao: Optional[Dict[str, Any]],
    input_dim: int,
    interpretabilidade: Dict[str, Any],
    caminho_interpretabilidade_run: str,
    caminho_relatorio_html_run: str,
    caminho_scored_dataset_run: str,
    postprocessor: FeaturePostProcessor,
    pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    return exportar_artefatos_execucao(
        model=model,
        pipeline=pipeline,
        task=task,
        modelo_final=modelo_final,
        run_context=run_context,
        caminhos_artefatos=caminhos_artefatos,
        config=config,
        metricas_treino=metricas_treino,
        metricas_teste=metricas_teste,
        threshold_classificacao=threshold_classificacao,
        best_params=best_params,
        best_cv_score=best_cv_score,
        resultado_otimizacao=resultado_otimizacao,
        model_train_context={"input_dim": input_dim, "output_dim": 1},
        postprocessor=postprocessor,
        pipeline_metadata_override=pipeline_metadata,
        interpretabilidade=interpretabilidade,
        caminho_interpretabilidade_run=caminho_interpretabilidade_run,
        caminho_relatorio_html_run=caminho_relatorio_html_run,
        caminho_scored_dataset_run=caminho_scored_dataset_run,
    )
