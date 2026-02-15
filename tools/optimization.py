import optuna
import numpy as np
from sklearn.model_selection import KFold


def otimizar_lightgbm(X, y, task="regression", n_trials=50):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        }

        if task == "regression":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)
        else:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**params)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(X):
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[val_idx], y[val_idx])
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
