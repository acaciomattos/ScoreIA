import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)


def metricas_regressao(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def otimizar_threshold_precision_recall(y_true, y_prob, beta=1.0):
    y_prob = np.asarray(y_prob).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns one extra point vs thresholds
    if thresholds.size == 0:
        return 0.5

    precisions = precisions[:-1]
    recalls = recalls[:-1]

    beta2 = beta ** 2
    denom = (beta2 * precisions) + recalls
    numer = (1.0 + beta2) * (precisions * recalls)
    fbeta = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=float),
        where=denom > 0,
    )

    best_idx = int(np.argmax(fbeta))
    return float(thresholds[best_idx])


def metricas_classificacao(y_true, y_pred, y_prob=None):
    auc = None

    if y_prob is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                auc_val = roc_auc_score(y_true, y_prob)

            if auc_val is not None and not np.isnan(auc_val):
                auc = float(auc_val)
        except Exception:
            auc = None

    return {
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "auc": auc,
    }
