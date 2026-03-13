import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, Optional

import joblib
import numpy as np

from tools.ml.artifacts import nome_artefato


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _dump_json(path: str, payload: Dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, ensure_ascii=False, indent=2)
    return path


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _coletar_integridade_artefatos(paths: Dict[str, Optional[str]]) -> Dict[str, Any]:
    artifacts = {}

    for key in sorted(paths.keys()):
        path = paths.get(key)
        if not path or not os.path.exists(path):
            continue

        artifacts[key] = {
            "path": path,
            "sha256": _sha256_file(path),
            "size_bytes": os.path.getsize(path),
        }

    assinatura_input = "\n".join(
        f"{k}:{v['sha256']}"
        for k, v in sorted(artifacts.items(), key=lambda item: item[0])
    )

    bundle_signature = hashlib.sha256(assinatura_input.encode("utf-8")).hexdigest()

    return {
        "algorithm": "sha256",
        "artifacts": artifacts,
        "bundle_signature_sha256": bundle_signature,
    }


def _coletar_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
    }

    modules = [
        "numpy",
        "scikit-learn",
        "lightgbm",
        "optuna",
        "torch",
        "polars",
        "joblib",
    ]

    for name in modules:
        try:
            if name == "scikit-learn":
                import sklearn  # type: ignore
                versions[name] = sklearn.__version__
            else:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not_available"

    return versions


def _serializar_otimizacao(resultado_otimizacao: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not resultado_otimizacao:
        return {"executada": False}

    if "melhor_modelo" in resultado_otimizacao:
        resumo = {
            "executada": True,
            "modo_auto": True,
            "melhor_modelo": resultado_otimizacao.get("melhor_modelo"),
            "best_score": resultado_otimizacao.get("resultado", {}).get("best_score"),
            "best_params": resultado_otimizacao.get("resultado", {}).get("best_params"),
            "todos": {},
        }

        for modelo, dados in resultado_otimizacao.get("todos", {}).items():
            resumo["todos"][modelo] = {
                "best_score": dados.get("best_score"),
                "best_params": dados.get("best_params"),
                "n_trials": len(dados.get("study").trials) if dados.get("study") is not None else 0,
            }

        return resumo

    return {
        "executada": True,
        "modo_auto": False,
        "best_score": resultado_otimizacao.get("best_score"),
        "best_params": resultado_otimizacao.get("best_params"),
        "n_trials": len(resultado_otimizacao.get("study").trials) if resultado_otimizacao.get("study") is not None else 0,
    }


def _escrever_script_inferencia(
    script_path: str,
    model_kind: str,
    model_file_name: str,
    preprocess_file_name: str,
    metadata_file_name: str,
) -> str:
    script = f"""import argparse
import json
import os

import joblib
import numpy as np
import polars as pl


def _transformar(df, preprocess):
    colunas_numericas = preprocess["colunas_numericas"]
    colunas_categoricas = preprocess["colunas_categoricas"]
    imputer_num = preprocess["imputer_num"]
    imputer_cat = preprocess["imputer_cat"]
    encoder = preprocess["encoder"]
    scaler = preprocess["scaler"]
    feature_order = preprocess["feature_order"]
    postprocessor = preprocess.get("postprocessor")
    segment_features = preprocess.get("segment_features") or []
    segment_bucket_col = preprocess.get("segment_bucket_col")

    if segment_bucket_col and segment_bucket_col in feature_order and segment_bucket_col not in df.columns and segment_features:
        bucket_values = [
            " | ".join([f"{{seg}}={{row.get(seg)}}" for seg in segment_features])
            for row in df.select(segment_features).to_dicts()
        ]
        df = df.with_columns(pl.Series(segment_bucket_col, bucket_values))

    X_df = df.select(feature_order)

    X_num = None
    X_cat = None

    if colunas_numericas:
        num_df = X_df.select(colunas_numericas).to_numpy()
        if imputer_num is not None:
            X_num = imputer_num.transform(num_df)
        else:
            X_num = num_df
        if scaler is not None:
            X_num = scaler.transform(X_num)

    if colunas_categoricas:
        cat_df = X_df.select(colunas_categoricas).to_numpy()
        if imputer_cat is not None:
            cat_df = imputer_cat.transform(cat_df)
        X_cat = encoder.transform(cat_df)

    if X_num is not None and X_cat is not None:
        X_out = np.hstack([X_num, X_cat])
    elif X_num is not None:
        X_out = X_num
    else:
        X_out = X_cat

    if postprocessor is not None:
        X_out = postprocessor.transform(X_out)

    return X_out


def _prepare_lightgbm_input(model, X):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = getattr(model, "feature_name_", None)
    if feature_names is None:
        return X

    cols = [str(col) for col in np.asarray(feature_names).tolist()]
    if len(cols) != X.shape[1]:
        return X

    return pl.DataFrame(X, schema=cols, orient="row")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.dirname(base_dir)
    model_dir = os.path.join(run_dir, "model")
    preprocess_dir = os.path.join(run_dir, "preprocess")
    metadata_dir = os.path.join(run_dir, "metadata")

    preprocess = joblib.load(os.path.join(preprocess_dir, "{preprocess_file_name}"))

    with open(os.path.join(metadata_dir, "{metadata_file_name}"), "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    if args.input_format == "csv":
        df = pl.read_csv(args.input)
    else:
        df = pl.read_parquet(args.input)

    X = _transformar(df, preprocess)

    expected_model_kind = "{model_kind}"
    model_kind = model_metadata.get("model_kind", expected_model_kind)

    if model_kind != expected_model_kind:
        print(
            f"[WARN] model_kind em metadata ({{model_kind}}) difere do artefato esperado ({{expected_model_kind}})."
        )

    task = model_metadata["task"]
    threshold = model_metadata.get("threshold_classificacao")

    if model_kind == "lightgbm":
        model = joblib.load(os.path.join(model_dir, "{model_file_name}"))
        X_model = _prepare_lightgbm_input(model, X)
        if task == "classification":
            y_prob = model.predict_proba(X_model)[:, 1]
            y_pred = (y_prob >= threshold).astype(int) if threshold is not None else (y_prob >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_model)
            y_prob = None
    elif model_kind == "lightgbm_bucket_ensemble":
        bundle = joblib.load(os.path.join(model_dir, "{model_file_name}"))
        segment_features = model_metadata.get("segment_features") or []
        bucket_keys = None
        if segment_features:
            bucket_keys = [
                " | ".join([f"{{seg}}={{row.get(seg)}}" for seg in segment_features])
                for row in df.select(segment_features).to_dicts()
            ]

        global_model = bundle["global_model"]
        bucket_models = bundle.get("bucket_models", {{}})

        if task == "classification":
            y_prob = np.zeros(X.shape[0], dtype=float)
            if bucket_keys is None:
                y_prob[:] = global_model.predict_proba(_prepare_lightgbm_input(global_model, X))[:, 1]
            else:
                bucket_keys = np.asarray(bucket_keys, dtype=object)
                for bucket in np.unique(bucket_keys):
                    idx = np.where(bucket_keys == bucket)[0]
                    bucket_model = bucket_models.get(str(bucket), global_model)
                    y_prob[idx] = bucket_model.predict_proba(_prepare_lightgbm_input(bucket_model, X[idx]))[:, 1]
            y_pred = (y_prob >= threshold).astype(int) if threshold is not None else (y_prob >= 0.5).astype(int)
        else:
            y_pred = np.zeros(X.shape[0], dtype=float)
            if bucket_keys is None:
                y_pred[:] = global_model.predict(_prepare_lightgbm_input(global_model, X))
            else:
                bucket_keys = np.asarray(bucket_keys, dtype=object)
                for bucket in np.unique(bucket_keys):
                    idx = np.where(bucket_keys == bucket)[0]
                    bucket_model = bucket_models.get(str(bucket), global_model)
                    y_pred[idx] = bucket_model.predict(_prepare_lightgbm_input(bucket_model, X[idx]))
            y_prob = None
    else:
        import torch
        from tools.ml.model_factory import ModelFactory

        payload = torch.load(os.path.join(model_dir, "{model_file_name}"), map_location="cpu")
        params = payload.get("params", {{}})
        input_dim = payload["input_dim"]
        output_dim = payload["output_dim"]
        model = ModelFactory.criar_mlp(input_dim=input_dim, output_dim=output_dim, params=params)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        with torch.no_grad():
            raw = model(torch.tensor(X, dtype=torch.float32)).numpy().squeeze()

        if task == "classification":
            y_prob = 1.0 / (1.0 + np.exp(-raw))
            y_pred = (y_prob >= threshold).astype(int) if threshold is not None else (y_prob >= 0.5).astype(int)
        else:
            y_pred = raw
            y_prob = None

    output_payload = {{"prediction": np.asarray(y_pred).tolist()}}
    if y_prob is not None:
        output_payload["probability"] = np.asarray(y_prob).tolist()
    pl.DataFrame(output_payload).write_csv(args.output)
    print(f"Predicoes salvas em: {{args.output}}")


if __name__ == "__main__":
    main()
"""

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    return script_path


def exportar_artefatos_execucao(
    model: Any,
    pipeline: Any,
    task: str,
    modelo_final: str,
    run_context: Dict[str, str],
    caminhos_artefatos: Dict[str, str],
    config: Dict[str, Any],
    metricas_treino: Dict[str, Any],
    metricas_teste: Dict[str, Any],
    threshold_classificacao: Optional[float] = None,
    best_params: Optional[Dict[str, Any]] = None,
    best_cv_score: Optional[float] = None,
    resultado_otimizacao: Optional[Dict[str, Any]] = None,
    model_train_context: Optional[Dict[str, Any]] = None,
    postprocessor: Optional[Any] = None,
    pipeline_metadata_override: Optional[Dict[str, Any]] = None,
    interpretabilidade: Optional[Dict[str, Any]] = None,
    caminho_interpretabilidade_run: Optional[str] = None,
    caminho_relatorio_html_run: Optional[str] = None,
    caminho_scored_dataset_run: Optional[str] = None,
) -> Dict[str, str]:
    ts = run_context["timestamp"]

    is_bucket_ensemble = isinstance(model, dict) and model.get("kind") == "bucket_ensemble"
    model_file_name = nome_artefato("model", run_context, "joblib" if (modelo_final == "lightgbm" or is_bucket_ensemble) else "pt")
    preprocess_file_name = nome_artefato("pipeline", run_context, "joblib")
    model_metadata_file = nome_artefato("model_metadata", run_context, "json")
    train_config_file = nome_artefato("train_config", run_context, "json")
    metrics_train_file = nome_artefato("metrics_train", run_context, "json")
    metrics_test_file = nome_artefato("metrics_test", run_context, "json")
    feature_schema_file = nome_artefato("feature_schema", run_context, "json")
    optuna_summary_file = nome_artefato("optuna_summary", run_context, "json")
    versions_file = nome_artefato("versions", run_context, "json")
    manifest_file = nome_artefato("artifact_manifest", run_context, "json")
    hashes_file = nome_artefato("artifact_hashes", run_context, "json")
    script_file = nome_artefato("predict", run_context, "py")
    infer_readme_file = nome_artefato("inference_readme", run_context, "txt")

    path_model = os.path.join(caminhos_artefatos["run_model_dir"], model_file_name)
    path_preprocess = os.path.join(caminhos_artefatos["run_preprocess_dir"], preprocess_file_name)
    path_model_metadata = os.path.join(caminhos_artefatos["run_metadata_dir"], model_metadata_file)
    path_train_config = os.path.join(caminhos_artefatos["run_metadata_dir"], train_config_file)
    path_metrics_train = os.path.join(caminhos_artefatos["run_metadata_dir"], metrics_train_file)
    path_metrics_test = os.path.join(caminhos_artefatos["run_metadata_dir"], metrics_test_file)
    path_feature_schema = os.path.join(caminhos_artefatos["run_metadata_dir"], feature_schema_file)
    path_optuna_summary = os.path.join(caminhos_artefatos["run_metadata_dir"], optuna_summary_file)
    path_versions = os.path.join(caminhos_artefatos["run_metadata_dir"], versions_file)
    path_manifest = os.path.join(caminhos_artefatos["run_metadata_dir"], manifest_file)
    path_hashes = os.path.join(caminhos_artefatos["run_metadata_dir"], hashes_file)
    path_script = os.path.join(caminhos_artefatos["run_inference_dir"], script_file)
    path_infer_readme = os.path.join(caminhos_artefatos["run_inference_dir"], infer_readme_file)

    if is_bucket_ensemble:
        joblib.dump(model, path_model)
        model_payload_context = {"model_kind": "lightgbm_bucket_ensemble"}
    elif modelo_final == "lightgbm":
        joblib.dump(model, path_model)
        model_payload_context = {"model_kind": "lightgbm"}
    else:
        import torch

        params = (best_params or {}).copy()
        ctx = model_train_context or {}
        torch.save(
            {
                "state_dict": model.state_dict(),
                "params": params,
                "input_dim": ctx.get("input_dim"),
                "output_dim": ctx.get("output_dim", 1),
            },
            path_model,
        )
        model_payload_context = {"model_kind": "mlp"}

    preprocess_bundle = {
        "target": pipeline.target,
        "feature_order": list(getattr(pipeline, "feature_order", pipeline.df.drop(pipeline.target).columns)),
        "colunas_numericas": pipeline.colunas_numericas,
        "colunas_categoricas": pipeline.colunas_categoricas,
        "imputer_num": pipeline.imputer_num,
        "imputer_cat": pipeline.imputer_cat,
        "encoder": pipeline.encoder,
        "scaler": pipeline.scaler,
        "feature_names_out": getattr(pipeline, "feature_names_out", []),
        "postprocessor": postprocessor,
        "segment_features": list(config.get("segment_features") or []),
        "segment_bucket_col": getattr(pipeline, "segment_bucket_col", None),
    }
    joblib.dump(preprocess_bundle, path_preprocess)

    postprocessor_metadata = {}
    if postprocessor is not None and hasattr(postprocessor, "metadata"):
        postprocessor_metadata = postprocessor.metadata()
    pipeline_metadata = pipeline_metadata_override or getattr(pipeline, "metadata", {})

    model_metadata = {
        "timestamp": ts,
        "task": task,
        "modelo_final": modelo_final,
        "threshold_classificacao": threshold_classificacao,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "postprocessor": postprocessor_metadata,
        "segment_features": list(config.get("segment_features") or []),
        "segment_strategy": config.get("segment_strategy"),
        **model_payload_context,
    }
    if model_train_context:
        model_metadata.update(model_train_context)
    _dump_json(path_model_metadata, model_metadata)

    _dump_json(path_train_config, {"config": config, "run_context": run_context})
    _dump_json(path_metrics_train, metricas_treino)
    _dump_json(path_metrics_test, metricas_teste)
    _dump_json(
        path_feature_schema,
        {
            "target": pipeline.target,
            "colunas_numericas": pipeline.colunas_numericas,
            "colunas_categoricas": pipeline.colunas_categoricas,
            "feature_names_out": getattr(pipeline, "feature_names_out", []),
            "postprocessor": postprocessor_metadata,
            "metadata": pipeline_metadata,
        },
    )
    _dump_json(path_optuna_summary, _serializar_otimizacao(resultado_otimizacao))
    _dump_json(path_versions, _coletar_versions())

    _escrever_script_inferencia(
        script_path=path_script,
        model_kind=model_payload_context["model_kind"],
        model_file_name=model_file_name,
        preprocess_file_name=preprocess_file_name,
        metadata_file_name=model_metadata_file,
    )

    with open(path_infer_readme, "w", encoding="utf-8") as f:
        f.write("Inference quickstart\n")
        f.write(f"Timestamp: {ts}\n")
        f.write("Example:\n")
        f.write(f"python {script_file} --input your_file.csv --input-format csv --output predictions.csv\n")

    if os.path.exists("requirements310.txt"):
        shutil.copy2("requirements310.txt", os.path.join(caminhos_artefatos["run_dir"], nome_artefato("requirements", run_context, "txt")))
    if os.path.exists("pyproject.toml"):
        shutil.copy2("pyproject.toml", os.path.join(caminhos_artefatos["run_dir"], nome_artefato("pyproject", run_context, "toml")))
    if os.path.exists("poetry.lock"):
        shutil.copy2("poetry.lock", os.path.join(caminhos_artefatos["run_dir"], nome_artefato("poetry", run_context, "lock")))

    caminho_interpretabilidade_final = None
    if caminho_interpretabilidade_run and os.path.exists(caminho_interpretabilidade_run):
        caminho_interpretabilidade_final = caminho_interpretabilidade_run
    elif interpretabilidade is not None:
        path_interp = os.path.join(
            caminhos_artefatos["run_reports_dir"],
            nome_artefato("interpretability", run_context, "json"),
        )
        _dump_json(path_interp, interpretabilidade)
        caminho_interpretabilidade_final = path_interp

    caminho_relatorio_final = None
    if caminho_relatorio_html_run and os.path.exists(caminho_relatorio_html_run):
        caminho_relatorio_final = caminho_relatorio_html_run

    integridade_payload = _coletar_integridade_artefatos(
        {
            "model": path_model,
            "preprocess": path_preprocess,
            "model_metadata": path_model_metadata,
            "train_config": path_train_config,
            "metrics_train": path_metrics_train,
            "metrics_test": path_metrics_test,
            "feature_schema": path_feature_schema,
            "optuna_summary": path_optuna_summary,
            "versions": path_versions,
            "inference_script": path_script,
            "inference_readme": path_infer_readme,
            "interpretability_report": caminho_interpretabilidade_final,
            "report_html": caminho_relatorio_final,
            "scored_dataset": caminho_scored_dataset_run,
        }
    )
    _dump_json(path_hashes, integridade_payload)

    manifest = {
        "timestamp": ts,
        "run_context": run_context,
        "paths": {
            "model": path_model,
            "preprocess": path_preprocess,
            "model_metadata": path_model_metadata,
            "train_config": path_train_config,
            "metrics_train": path_metrics_train,
            "metrics_test": path_metrics_test,
            "feature_schema": path_feature_schema,
            "optuna_summary": path_optuna_summary,
            "versions": path_versions,
            "inference_script": path_script,
            "inference_readme": path_infer_readme,
            "artifact_hashes": path_hashes,
            "interpretability_report": caminho_interpretabilidade_final,
            "report_html": caminho_relatorio_final,
            "scored_dataset": caminho_scored_dataset_run,
        },
        "integrity": {
            "algorithm": integridade_payload.get("algorithm"),
            "bundle_signature_sha256": integridade_payload.get("bundle_signature_sha256"),
            "artifacts_count": len(integridade_payload.get("artifacts", {})),
        },
    }
    _dump_json(path_manifest, manifest)

    return {
        "model": path_model,
        "preprocess": path_preprocess,
        "model_metadata": path_model_metadata,
        "train_config": path_train_config,
        "metrics_train": path_metrics_train,
        "metrics_test": path_metrics_test,
        "feature_schema": path_feature_schema,
        "optuna_summary": path_optuna_summary,
        "versions": path_versions,
        "inference_script": path_script,
        "inference_readme": path_infer_readme,
        "artifact_hashes": path_hashes,
        "manifest": path_manifest,
        "bundle_signature_sha256": integridade_payload.get("bundle_signature_sha256", ""),
        "interpretability_report": caminho_interpretabilidade_final or "",
        "report_html": caminho_relatorio_final or "",
        "scored_dataset": caminho_scored_dataset_run or "",
    }












