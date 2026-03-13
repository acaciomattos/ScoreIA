import argparse
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
            " | ".join([f"{seg}={row.get(seg)}" for seg in segment_features])
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

    preprocess = joblib.load(os.path.join(preprocess_dir, "pipeline_20260309192613.joblib"))

    with open(os.path.join(metadata_dir, "model_metadata_20260309192613.json"), "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    if args.input_format == "csv":
        df = pl.read_csv(args.input)
    else:
        df = pl.read_parquet(args.input)

    X = _transformar(df, preprocess)

    expected_model_kind = "lightgbm"
    model_kind = model_metadata.get("model_kind", expected_model_kind)

    if model_kind != expected_model_kind:
        print(
            f"[WARN] model_kind em metadata ({model_kind}) difere do artefato esperado ({expected_model_kind})."
        )

    task = model_metadata["task"]
    threshold = model_metadata.get("threshold_classificacao")

    if model_kind == "lightgbm":
        model = joblib.load(os.path.join(model_dir, "model_20260309192613.joblib"))
        X_model = _prepare_lightgbm_input(model, X)
        if task == "classification":
            y_prob = model.predict_proba(X_model)[:, 1]
            y_pred = (y_prob >= threshold).astype(int) if threshold is not None else (y_prob >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_model)
            y_prob = None
    elif model_kind == "lightgbm_bucket_ensemble":
        bundle = joblib.load(os.path.join(model_dir, "model_20260309192613.joblib"))
        segment_features = model_metadata.get("segment_features") or []
        bucket_keys = None
        if segment_features:
            bucket_keys = [
                " | ".join([f"{seg}={row.get(seg)}" for seg in segment_features])
                for row in df.select(segment_features).to_dicts()
            ]

        global_model = bundle["global_model"]
        bucket_models = bundle.get("bucket_models", {})

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
        from tools.model_factory import ModelFactory

        payload = torch.load(os.path.join(model_dir, "model_20260309192613.joblib"), map_location="cpu")
        params = payload.get("params", {})
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

    output_payload = {"prediction": np.asarray(y_pred).tolist()}
    if y_prob is not None:
        output_payload["probability"] = np.asarray(y_prob).tolist()
    pl.DataFrame(output_payload).write_csv(args.output)
    print(f"Predicoes salvas em: {args.output}")


if __name__ == "__main__":
    main()
