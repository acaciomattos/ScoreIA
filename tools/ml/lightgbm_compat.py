from typing import Any

import numpy as np
import polars as pl


def prepare_lightgbm_predict_input(model: Any, X: Any) -> Any:
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        return X_arr

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = getattr(model, "feature_name_", None)

    if feature_names is None:
        return X_arr

    columns = [str(col) for col in np.asarray(feature_names).tolist()]
    if len(columns) != X_arr.shape[1]:
        return X_arr

    return pl.DataFrame(X_arr, schema=columns, orient="row")
