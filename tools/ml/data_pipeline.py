# data_pipeline.py

from datetime import datetime

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPipeline:

    def __init__(self, caminho):
        self.caminho = caminho
        self.df = None
        self.target = None

        self.colunas_numericas = []
        self.colunas_categoricas = []

        self.imputer_num = None
        self.imputer_cat = None
        self.encoder = None
        self.scaler = None
        self.aplicar_tratamento_missing = True

        self.metadata = {}
        self.feature_names_out = []
        self.feature_order = []
        self.segment_bucket_col = "__scoreia_segment_bucket__"

    # ==========================================================
    # CARREGAMENTO
    # ==========================================================

    def carregar(self, formato):

        if formato == "csv":
            self.df = pl.read_csv(self.caminho)

        elif formato == "parquet":
            self.df = pl.read_parquet(self.caminho)

        else:
            raise ValueError("Formato nao suportado.")

        return self.df

    # ==========================================================
    # INFERENCIA DE TIPOS
    # ==========================================================

    def _inferir_tipos(self, X_df):

        self.colunas_numericas = []
        self.colunas_categoricas = []

        for col, dtype in zip(X_df.columns, X_df.dtypes):

            if hasattr(dtype, "is_numeric"):
                eh_numerica = bool(dtype.is_numeric())
            elif hasattr(pl.datatypes, "is_numeric"):
                eh_numerica = bool(pl.datatypes.is_numeric(dtype))
            else:
                eh_numerica = False

            if eh_numerica:
                self.colunas_numericas.append(col)
            else:
                self.colunas_categoricas.append(col)

    # ==========================================================
    # TEMPORAL: VALIDACAO
    # ==========================================================

    @staticmethod
    def _parse_month_key(value):
        if value is None:
            return None, None

        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan"}:
            return None, None

        if text.isdigit() and len(text) == 8:
            try:
                datetime.strptime(text, "%Y%m%d")
                return text[:6], "YYYYMMDD"
            except ValueError:
                return None, None

        if text.isdigit() and len(text) == 6:
            try:
                datetime.strptime(text, "%Y%m")
                return text, "YYYYMM"
            except ValueError:
                return None, None

        return None, None

    def _extrair_temporal_info(self, temporal_feature):
        if self.df is None:
            raise ValueError("Dados nao carregados.")

        if temporal_feature not in self.df.columns:
            raise ValueError(f"Feature temporal '{temporal_feature}' nao encontrada na base.")

        values = self.df[temporal_feature].to_list()
        month_keys = []
        formatos = set()
        invalidos = []

        for idx, raw in enumerate(values):
            month_key, formato = self._parse_month_key(raw)
            month_keys.append(month_key)

            if month_key is None:
                invalidos.append({"row_index": int(idx), "value": raw})
            else:
                formatos.add(formato)

        return {
            "month_keys": month_keys,
            "formats_detected": sorted([f for f in formatos if f]),
            "invalid_rows": invalidos,
        }

    def validar_temporal_feature(self, temporal_feature):
        info = self._extrair_temporal_info(temporal_feature)
        invalidos = info["invalid_rows"]

        if invalidos:
            amostras = ", ".join([repr(item["value"]) for item in invalidos[:5]])
            return {
                "valid": False,
                "error": (
                    f"A feature temporal '{temporal_feature}' possui valores fora do formato esperado "
                    f"(YYYYMMDD ou YYYYMM). Exemplos invalidos: {amostras}."
                ),
                "n_invalid_rows": len(invalidos),
                "months_sorted": [],
                "n_months": 0,
                "formats_detected": info["formats_detected"],
            }

        months_sorted = sorted(set(info["month_keys"]))
        return {
            "valid": True,
            "error": None,
            "n_invalid_rows": 0,
            "months_sorted": months_sorted,
            "n_months": len(months_sorted),
            "formats_detected": info["formats_detected"],
        }

    @staticmethod
    def build_segment_bucket_values(df, segment_features):
        segment_features = [s for s in (segment_features or []) if s in df.columns]
        if not segment_features:
            return []

        rows = df.select(segment_features).to_dicts()
        return [
            " | ".join([f"{seg}={row.get(seg)}" for seg in segment_features])
            for row in rows
        ]

    # ==========================================================
    # SPLIT HOLDOUT (ANTES DE QUALQUER FIT)
    # ==========================================================

    def split_holdout(
        self,
        target,
        test_size=0.2,
        task="regression",
        temporal_feature=None,
        oot_months=0,
        segment_features=None,
        segment_strategy="global",
    ):

        if self.df is None:
            raise ValueError("Dados nao carregados.")

        if target not in self.df.columns:
            raise ValueError("Target nao encontrado.")

        self.target = target

        segment_features = [s for s in (segment_features or []) if s in self.df.columns]
        oot_months = int(oot_months or 0)
        segment_strategy = str(segment_strategy or "global")

        temporal_validation = None
        temporal_month_col = "__scoreia_temporal_month__"
        df_base = self.df
        oot_month_list = []

        if temporal_feature:
            temporal_validation = self.validar_temporal_feature(temporal_feature)
            if not temporal_validation["valid"]:
                raise ValueError(temporal_validation["error"])

            month_keys = self._extrair_temporal_info(temporal_feature)["month_keys"]
            df_base = self.df.with_columns(pl.Series(temporal_month_col, month_keys))

            if oot_months > 0:
                months = temporal_validation["months_sorted"]
                if len(months) < 2:
                    raise ValueError("Sao necessarios ao menos 2 meses distintos para separar OOT.")
                if oot_months >= len(months):
                    raise ValueError(
                        f"oot_months={oot_months} invalido: deve ser menor que o total de meses ({len(months)})."
                    )
                oot_month_list = months[-oot_months:]

        if segment_features:
            segment_bucket_values = self.build_segment_bucket_values(df_base, segment_features)
            df_base = df_base.with_columns(pl.Series(self.segment_bucket_col, segment_bucket_values))

        feature_order = [c for c in df_base.columns if c != target]
        if temporal_feature and temporal_feature in feature_order:
            feature_order = [c for c in feature_order if c != temporal_feature]
        if temporal_month_col in feature_order:
            feature_order = [c for c in feature_order if c != temporal_month_col]
        if segment_strategy == "per_bucket_model":
            feature_order = [c for c in feature_order if c not in set(segment_features)]
            if self.segment_bucket_col in feature_order:
                feature_order = [c for c in feature_order if c != self.segment_bucket_col]
        elif segment_strategy == "bucket_feature" and segment_features and self.segment_bucket_col not in feature_order:
            feature_order.append(self.segment_bucket_col)

        self.feature_order = feature_order
        self._inferir_tipos(df_base.select(self.feature_order))

        if oot_month_list:
            df_oot = df_base.filter(pl.col(temporal_month_col).is_in(oot_month_list))
            df_in_time = df_base.filter(~pl.col(temporal_month_col).is_in(oot_month_list))
        else:
            df_oot = pl.DataFrame(schema=df_base.schema)
            df_in_time = df_base

        if df_in_time.height < 2:
            raise ValueError("Volume insuficiente no periodo in-time para separar treino/teste.")

        df_in_time = df_in_time.with_row_index("__row_id__")
        y_in_time = df_in_time[target].to_numpy()
        row_ids = df_in_time["__row_id__"].to_numpy()

        stratify = y_in_time if task == "classification" else None
        row_train, row_test = train_test_split(
            row_ids,
            test_size=test_size,
            stratify=stratify,
            random_state=42,
            shuffle=True,
        )

        row_train_list = row_train.tolist() if hasattr(row_train, "tolist") else list(row_train)
        row_test_list = row_test.tolist() if hasattr(row_test, "tolist") else list(row_test)

        df_train = df_in_time.filter(pl.col("__row_id__").is_in(row_train_list)).drop("__row_id__")
        df_test = df_in_time.filter(pl.col("__row_id__").is_in(row_test_list)).drop("__row_id__")

        X_train = df_train.select(self.feature_order).to_numpy()
        y_train = df_train[target].to_numpy()

        X_test = df_test.select(self.feature_order).to_numpy()
        y_test = df_test[target].to_numpy()

        if df_oot.height > 0:
            X_oot = df_oot.select(self.feature_order).to_numpy()
            y_oot = df_oot[target].to_numpy()
        else:
            n_features = len(self.feature_order)
            X_oot = np.empty((0, n_features))
            y_oot = np.asarray([])

        split_info = {
            "temporal_feature": temporal_feature,
            "temporal_month_col": temporal_month_col if temporal_feature else None,
            "temporal_validation": temporal_validation,
            "oot_enabled": bool(oot_month_list),
            "oot_months": oot_months if oot_month_list else 0,
            "oot_month_list": oot_month_list,
            "excluded_from_model": [temporal_feature] if temporal_feature else [],
            "segment_features": segment_features,
            "segment_strategy": segment_strategy,
            "segment_bucket_col": self.segment_bucket_col if segment_features else None,
            "df_train_raw": df_train,
            "df_test_raw": df_test,
            "df_oot_raw": df_oot,
            "segment_bucket_train": df_train[self.segment_bucket_col].to_list() if segment_features and self.segment_bucket_col in df_train.columns else [],
            "segment_bucket_test": df_test[self.segment_bucket_col].to_list() if segment_features and self.segment_bucket_col in df_test.columns else [],
            "segment_bucket_oot": df_oot[self.segment_bucket_col].to_list() if segment_features and self.segment_bucket_col in df_oot.columns else [],
        }

        return X_train, X_test, y_train, y_test, X_oot, y_oot, split_info

    # ==========================================================
    # FIT TRANSFORM (APENAS TREINO)
    # ==========================================================

    def fit_transform(self, X_train, modelo, aplicar_tratamento_missing=True):

        X_train_df = pl.DataFrame(X_train, schema=self.feature_order)
        self.aplicar_tratamento_missing = bool(aplicar_tratamento_missing)

        # ---------------- NUMERICAS ----------------
        X_num = None
        nomes_num = []

        if self.colunas_numericas:
            num_df = X_train_df.select(self.colunas_numericas).to_numpy()

            if self.aplicar_tratamento_missing:
                self.imputer_num = SimpleImputer(strategy="median")
                num_df = self.imputer_num.fit_transform(num_df)
            else:
                self.imputer_num = None

            if modelo == "mlp":
                if np.isnan(np.asarray(num_df, dtype=float)).any():
                    raise ValueError(
                        "MLP requer tratamento de missing nas variaveis numericas. Ative a opcao de imputacao."
                    )
                self.scaler = StandardScaler()
                num_df = self.scaler.fit_transform(num_df)

            X_num = num_df
            nomes_num = list(self.colunas_numericas)

        # ---------------- CATEGORICAS ----------------
        X_cat = None
        nomes_cat = []

        if self.colunas_categoricas:
            cat_df = X_train_df.select(self.colunas_categoricas).to_numpy()

            if self.aplicar_tratamento_missing:
                self.imputer_cat = SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                )
                cat_df = self.imputer_cat.fit_transform(cat_df)
            else:
                self.imputer_cat = None

            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )

            X_cat = self.encoder.fit_transform(cat_df)
            nomes_cat = self.encoder.get_feature_names_out(self.colunas_categoricas).tolist()

        # ---------------- CONCAT ----------------
        if X_num is not None and X_cat is not None:
            X_final = np.hstack([X_num, X_cat])
        elif X_num is not None:
            X_final = X_num
        else:
            X_final = X_cat

        self.feature_names_out = nomes_num + nomes_cat

        self.metadata = {
            "n_features": X_final.shape[1],
            "n_numericas": len(self.colunas_numericas),
            "n_categoricas": len(self.colunas_categoricas),
            "feature_names_out": self.feature_names_out,
            "feature_order_model": list(self.feature_order),
            "aplicar_tratamento_missing": bool(self.aplicar_tratamento_missing),
        }

        return X_final

    # ==========================================================
    # TRANSFORM (TESTE OU NOVOS DADOS)
    # ==========================================================

    def transform(self, X):

        X_df = pl.DataFrame(X, schema=self.feature_order)

        X_num = None
        X_cat = None

        if self.colunas_numericas:
            num_df = X_df.select(self.colunas_numericas).to_numpy()
            if self.imputer_num is not None:
                num_df = self.imputer_num.transform(num_df)

            if self.scaler:
                num_df = self.scaler.transform(num_df)

            X_num = num_df

        if self.colunas_categoricas:
            cat_df = X_df.select(self.colunas_categoricas).to_numpy()
            if self.imputer_cat is not None:
                cat_df = self.imputer_cat.transform(cat_df)
            X_cat = self.encoder.transform(cat_df)

        if X_num is not None and X_cat is not None:
            return np.hstack([X_num, X_cat])
        if X_num is not None:
            return X_num
        return X_cat
