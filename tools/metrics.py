from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score
)
import numpy as np


def metricas_regressao(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }


def metricas_classificacao(y_true, y_pred, y_prob=None):
    return {
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None
    }
