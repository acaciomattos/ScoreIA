import optuna
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from tools.model_factory import ModelFactory
from tools.metrics import (
    metricas_regressao,
    metricas_classificacao
)


# ==========================================================
# UTILITÁRIOS
# ==========================================================

def _get_metric_key(task):
    return "rmse" if task == "regression" else "auc"


def _direction(task):
    return "minimize" if task == "regression" else "maximize"


# ==========================================================
# TREINO MLP COM EARLY STOPPING
# ==========================================================

def _treinar_mlp(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    task,
    params,
    trial=None,
    patience=20,
    max_epochs=300,
    batch_size=64
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get("lr", 1e-3)
    )

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_loss = np.inf
    best_state = None
    counter = 0

    for epoch in range(max_epochs):

        model.train()
        optimizer.zero_grad()

        outputs = model(X_train).squeeze()

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validação
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val).item()

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if counter >= patience:
            break

    model.load_state_dict(best_state)

    return model


# ==========================================================
# OBJETIVO PARA CADA MODELO
# ==========================================================

def _objective_factory(folds, task, modelo, early_stopping_rounds):

    metric_key = _get_metric_key(task)

    def objective(trial):

        # ==========================================
        # HIPERPARÂMETROS
        # ==========================================

        if modelo == "lightgbm":

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "n_estimators": 5000
            }

        elif modelo == "mlp":

            params = {
                "hidden_units": trial.suggest_int("hidden_units", 32, 256),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            }

        else:
            raise ValueError("Modelo não suportado.")

        scores = []

        # ==========================================
        # CROSS VALIDATION REAL
        # ==========================================

        for X_train, X_val, y_train, y_val in folds:

            if modelo == "lightgbm":

                model = ModelFactory.criar_lightgbm(
                    task=task,
                    params=params
                )

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=metric_key,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )

                y_pred = model.predict(X_val)

            else:  # MLP

                input_dim = X_train.shape[1]
                output_dim = 1

                model = ModelFactory.criar_mlp(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    params=params
                )

                model = _treinar_mlp(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    task,
                    params,
                    trial=trial
                )

                with torch.no_grad():
                    preds = model(
                        torch.tensor(X_val, dtype=torch.float32)
                    ).numpy()

                y_pred = preds.squeeze()

            if task == "regression":
                metricas = metricas_regressao(y_val, y_pred)
                scores.append(metricas[metric_key])
            else:
                metricas = metricas_classificacao(y_val, y_pred)
                scores.append(metricas[metric_key])

        return np.mean(scores)

    return objective


# ==========================================================
# OTIMIZAÇÃO PRINCIPAL
# ==========================================================

def otimizar_modelo_cv(
    folds,
    task,
    modelo="lightgbm",
    n_trials=50,
    early_stopping_rounds=50
):

    modelos = ["lightgbm", "mlp"] if modelo == "auto" else [modelo]

    resultados = {}

    for m in modelos:

        study = optuna.create_study(direction=_direction(task))

        objective = _objective_factory(
            folds=folds,
            task=task,
            modelo=m,
            early_stopping_rounds=early_stopping_rounds
        )

        study.optimize(objective, n_trials=n_trials)

        resultados[m] = {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study
        }

    # Se auto, retorna o melhor modelo
    if modelo == "auto":

        if task == "regression":
            melhor = min(resultados.items(), key=lambda x: x[1]["best_score"])
        else:
            melhor = max(resultados.items(), key=lambda x: x[1]["best_score"])

        return {
            "melhor_modelo": melhor[0],
            "resultado": melhor[1],
            "todos": resultados
        }

    return resultados[modelo]
