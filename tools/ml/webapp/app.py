import os

from flask import Flask, render_template, request

from core.agent import Agente
from tools.ml.webapp.storage import resolver_dataset_web
from tools.ml.webapp.validation import validar_temporal_oot_web


app = Flask(__name__)
agente = Agente()


def _parse_int(value, default):
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _resolve_form_action(form):
    return (
        (form.get("form_action") or form.get("submit_action") or "train_pipeline")
        .strip()
    )


def _resolve_persisted_dataset_name(file_storage, persisted_dataset_name, persisted_dataset_path):
    uploaded_name = (getattr(file_storage, "filename", "") or "").strip()
    if uploaded_name:
        return uploaded_name
    if persisted_dataset_name:
        return persisted_dataset_name
    if persisted_dataset_path:
        return os.path.basename(persisted_dataset_path)
    return ""


def _build_form_state(form, persisted_dataset_path, persisted_dataset_name=""):
    segment_features = [
        s.strip()
        for s in form.getlist("segment_features[]")
        if s and s.strip()
    ]

    return {
        "persisted_dataset_path": persisted_dataset_path,
        "persisted_dataset_name": persisted_dataset_name,
        "target": (form.get("target") or "").strip(),
        "task": form.get("task", "classification"),
        "formato": form.get("formato", "csv"),
        "modelo": form.get("modelo", "lightgbm"),
        "otimizar": form.get("otimizar") == "on",
        "n_trials": _parse_int(form.get("n_trials"), 30),
        "early_stopping_rounds": _parse_int(form.get("early_stopping_rounds"), 20),
        "apply_missing_treatment": form.get("apply_missing_treatment") == "on",
        "prune_best_model": form.get("prune_best_model") == "on",
        "prune_importance_cutoff_pct": form.get("prune_importance_cutoff_pct") or "80",
        "reduce_tail_features": form.get("reduce_tail_features") == "on",
        "reduction_tail_features_count": _parse_int(form.get("reduction_tail_features_count"), 10),
        "temporal_feature": (form.get("temporal_feature") or "").strip() or None,
        "segment_features": segment_features,
        "segment_strategy": form.get("segment_strategy", "global"),
        "oot_months": (form.get("oot_months") or "").strip(),
        "stability_feature_selection": form.get("stability_feature_selection") == "on",
        "stability_max_folds": (form.get("stability_max_folds") or "").strip(),
        "stability_max_candidates": _parse_int(form.get("stability_max_candidates"), 8),
        "stability_perf_rel_tolerance": form.get("stability_perf_rel_tolerance") or "0.05",
        "stability_perf_abs_tolerance": (form.get("stability_perf_abs_tolerance") or "").strip(),
        "stability_importance_cutoff_pct": form.get("stability_importance_cutoff_pct") or "80",
        "classification_balance_strategy": form.get("classification_balance_strategy", "none"),
        "positive_class_weight": (form.get("positive_class_weight") or "").strip(),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_action = _resolve_form_action(request.form)
        formato = request.form.get("formato", "csv")
        arquivo = request.files.get("file")
        persisted_dataset_path_in = (request.form.get("persisted_dataset_path") or "").strip() or None
        persisted_dataset_name_in = (request.form.get("persisted_dataset_name") or "").strip() or None

        try:
            caminho_dataset = resolver_dataset_web(arquivo, persisted_dataset_path_in, formato)
        except Exception as exc:
            return render_template(
                "index.html",
                erro_execucao=str(exc),
                config_enviada=_build_form_state(
                    request.form,
                    persisted_dataset_path_in,
                    persisted_dataset_name_in or "",
                ),
                persisted_dataset_path=persisted_dataset_path_in,
                persisted_dataset_name=persisted_dataset_name_in,
            )

        persisted_dataset_name = _resolve_persisted_dataset_name(
            arquivo,
            persisted_dataset_name_in,
            caminho_dataset,
        )
        config_enviada = _build_form_state(
            request.form,
            caminho_dataset,
            persisted_dataset_name,
        )

        try:
            validacao_temporal = validar_temporal_oot_web(
                caminho=caminho_dataset,
                formato=formato,
                temporal_feature=config_enviada["temporal_feature"],
                oot_months_raw=config_enviada["oot_months"],
                stability_max_folds_raw=config_enviada["stability_max_folds"],
            )

            temporal_validation = validacao_temporal["temporal_validation"]
            stability_fold_bounds = validacao_temporal["stability_fold_bounds"]
            show_oot_field = validacao_temporal["show_oot_field"]

            if form_action == "validate_only":
                alerta_temporal = validacao_temporal["alerta_temporal"]
                if not alerta_temporal and not validacao_temporal["erro_temporal"]:
                    if temporal_validation:
                        alerta_temporal = (
                            "Dataset persistido e feature temporal validada. "
                            f"Meses distintos encontrados: {temporal_validation['n_months']}."
                        )
                    else:
                        alerta_temporal = "Dataset persistido para reutilizacao. Nenhuma feature temporal foi informada."

                return render_template(
                    "index.html",
                    alerta_temporal=alerta_temporal,
                    erro_temporal=validacao_temporal["erro_temporal"],
                    temporal_validation=temporal_validation,
                    stability_fold_bounds=stability_fold_bounds,
                    config_enviada=config_enviada,
                    show_oot_field=show_oot_field,
                    persisted_dataset_path=caminho_dataset,
                    persisted_dataset_name=persisted_dataset_name,
                )

            if not config_enviada["target"]:
                raise ValueError("O parametro 'target' deve ser informado para treinar o pipeline.")

            if not validacao_temporal["is_valid_for_training"]:
                return render_template(
                    "index.html",
                    erro_temporal=validacao_temporal["erro_temporal"],
                    alerta_temporal=validacao_temporal["alerta_temporal"],
                    temporal_validation=temporal_validation,
                    stability_fold_bounds=stability_fold_bounds,
                    config_enviada=config_enviada,
                    show_oot_field=show_oot_field,
                    persisted_dataset_path=caminho_dataset,
                    persisted_dataset_name=persisted_dataset_name,
                )

            if config_enviada["stability_feature_selection"]:
                stability_max_folds = validacao_temporal["stability_max_folds_resolved"] or (
                    stability_fold_bounds["default"] if stability_fold_bounds else 6
                )
            else:
                stability_max_folds = _parse_int(config_enviada["stability_max_folds"], 6)

            config = {
                "caminho": caminho_dataset,
                "formato": config_enviada["formato"],
                "target": config_enviada["target"],
                "task": config_enviada["task"],
                "modelo": config_enviada["modelo"],
                "otimizar": config_enviada["otimizar"],
                "n_trials": config_enviada["n_trials"],
                "early_stopping_rounds": config_enviada["early_stopping_rounds"],
                "apply_missing_treatment": config_enviada["apply_missing_treatment"],
                "prune_best_model": config_enviada["prune_best_model"],
                "prune_importance_cutoff_pct": float(config_enviada["prune_importance_cutoff_pct"]),
                "reduce_tail_features": config_enviada["reduce_tail_features"],
                "reduction_tail_features_count": config_enviada["reduction_tail_features_count"],
                "reduction_variance_threshold": 0.70,
                "temporal_feature": config_enviada["temporal_feature"],
                "segment_features": config_enviada["segment_features"],
                "segment_strategy": config_enviada["segment_strategy"],
                "oot_months": validacao_temporal["oot_months"],
                "stability_feature_selection": config_enviada["stability_feature_selection"],
                "stability_max_folds": stability_max_folds,
                "stability_max_candidates": config_enviada["stability_max_candidates"],
                "stability_perf_rel_tolerance": float(config_enviada["stability_perf_rel_tolerance"]),
                "stability_perf_abs_tolerance": _parse_optional_float(config_enviada["stability_perf_abs_tolerance"]),
                "stability_importance_cutoff_pct": float(config_enviada["stability_importance_cutoff_pct"]),
                "classification_balance_strategy": config_enviada["classification_balance_strategy"],
                "positive_class_weight": _parse_optional_float(config_enviada["positive_class_weight"]),
            }

            resultado = agente.analisar_dados(config)

            return render_template(
                "index.html",
                resultado=resultado,
                temporal_validation=temporal_validation,
                stability_fold_bounds=stability_fold_bounds,
                config_enviada=config_enviada,
                alerta_temporal=(
                    f"Feature temporal validada. Meses distintos encontrados: {temporal_validation['n_months']}."
                    if temporal_validation
                    else None
                ),
                show_oot_field=show_oot_field,
                persisted_dataset_path=caminho_dataset,
                persisted_dataset_name=persisted_dataset_name,
            )
        except Exception as exc:
            return render_template(
                "index.html",
                erro_execucao=str(exc),
                temporal_validation=validacao_temporal.get("temporal_validation") if "validacao_temporal" in locals() else None,
                stability_fold_bounds=validacao_temporal.get("stability_fold_bounds") if "validacao_temporal" in locals() else None,
                config_enviada=config_enviada,
                show_oot_field=bool(config_enviada["temporal_feature"]),
                persisted_dataset_path=caminho_dataset,
                persisted_dataset_name=persisted_dataset_name,
            )

    return render_template(
        "index.html",
        config_enviada={
            "task": "classification",
            "formato": "csv",
            "modelo": "lightgbm",
            "otimizar": True,
            "n_trials": 30,
            "early_stopping_rounds": 20,
            "apply_missing_treatment": True,
            "prune_best_model": False,
            "prune_importance_cutoff_pct": "80",
            "reduce_tail_features": False,
            "reduction_tail_features_count": 10,
            "segment_features": [],
            "segment_strategy": "global",
            "stability_feature_selection": True,
            "stability_max_candidates": 8,
            "stability_perf_rel_tolerance": "0.05",
            "stability_perf_abs_tolerance": "",
            "stability_importance_cutoff_pct": "80",
            "stability_max_folds": "",
            "oot_months": "",
            "classification_balance_strategy": "none",
            "positive_class_weight": "",
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
