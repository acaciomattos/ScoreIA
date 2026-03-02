# data_pipeline.py

import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


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

        self.metadata = {}

    # ==========================================================
    # CARREGAMENTO
    # ==========================================================

    def carregar(self, formato):

        if formato == "csv":
            self.df = pl.read_csv(self.caminho)

        elif formato == "parquet":
            self.df = pl.read_parquet(self.caminho)

        else:
            raise ValueError("Formato não suportado.")

        return self.df

    # ==========================================================
    # INFERÊNCIA DE TIPOS
    # ==========================================================

    def _inferir_tipos(self, X_df):

        self.colunas_numericas = []
        self.colunas_categoricas = []

        for col, dtype in zip(X_df.columns, X_df.dtypes):

            if pl.datatypes.is_numeric(dtype):
                self.colunas_numericas.append(col)
            else:
                self.colunas_categoricas.append(col)

    # ==========================================================
    # SPLIT HOLDOUT (ANTES DE QUALQUER FIT)
    # ==========================================================

    def split_holdout(self, target, test_size=0.2, task="regression"):

        if self.df is None:
            raise ValueError("Dados não carregados.")

        if target not in self.df.columns:
            raise ValueError("Target não encontrado.")

        self.target = target

        y = self.df[target].to_numpy()
        X_df = self.df.drop(target)

        self._inferir_tipos(X_df)

        X = X_df.to_numpy()

        stratify = y if task == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=stratify,
            random_state=42,
            shuffle=True
        )

        return X_train, X_test, y_train, y_test

    # ==========================================================
    # FIT TRANSFORM (APENAS TREINO)
    # ==========================================================

    def fit_transform(self, X_train, modelo):

        X_train_df = pl.DataFrame(X_train, schema=self.df.drop(self.target).columns)

        # ---------------- NUMÉRICAS ----------------
        X_num = None

        if self.colunas_numericas:
            num_df = X_train_df.select(self.colunas_numericas).to_pandas()

            self.imputer_num = SimpleImputer(strategy="median")
            num_df = self.imputer_num.fit_transform(num_df)

            if modelo == "mlp":
                self.scaler = StandardScaler()
                num_df = self.scaler.fit_transform(num_df)

            X_num = num_df

        # ---------------- CATEGÓRICAS ----------------
        X_cat = None

        if self.colunas_categoricas:
            cat_df = X_train_df.select(self.colunas_categoricas).to_pandas()

            self.imputer_cat = SimpleImputer(
                strategy="constant",
                fill_value="missing"
            )

            cat_df = self.imputer_cat.fit_transform(cat_df)

            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

            X_cat = self.encoder.fit_transform(cat_df)

        # ---------------- CONCAT ----------------
        if X_num is not None and X_cat is not None:
            X_final = np.hstack([X_num, X_cat])
        elif X_num is not None:
            X_final = X_num
        else:
            X_final = X_cat

        self.metadata = {
            "n_features": X_final.shape[1],
            "n_numericas": len(self.colunas_numericas),
            "n_categoricas": len(self.colunas_categoricas)
        }

        return X_final

    # ==========================================================
    # TRANSFORM (TESTE OU NOVOS DADOS)
    # ==========================================================

    def transform(self, X):

        X_df = pl.DataFrame(X, schema=self.df.drop(self.target).columns)

        X_num = None
        X_cat = None

        if self.colunas_numericas:
            num_df = X_df.select(self.colunas_numericas).to_pandas()
            num_df = self.imputer_num.transform(num_df)

            if self.scaler:
                num_df = self.scaler.transform(num_df)

            X_num = num_df

        if self.colunas_categoricas:
            cat_df = X_df.select(self.colunas_categoricas).to_pandas()
            cat_df = self.imputer_cat.transform(cat_df)
            X_cat = self.encoder.transform(cat_df)

        if X_num is not None and X_cat is not None:
            return np.hstack([X_num, X_cat])
        elif X_num is not None:
            return X_num
        else:
            return X_cat
