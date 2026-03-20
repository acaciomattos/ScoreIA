import lightgbm as lgb
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_units=128,
        n_layers=2,
        dropout=0.0
    ):
        super().__init__()

        layers = []
        in_features = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = hidden_units

        layers.append(nn.Linear(in_features, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ModelFactory:

    @staticmethod
    def criar_lightgbm(task="regression", params=None):

        params = params or {}

        if task == "regression":
            return lgb.LGBMRegressor(**params)

        elif task == "classification":
            return lgb.LGBMClassifier(**params)

        else:
            raise ValueError("Task inválida.")

    @staticmethod
    def criar_mlp(input_dim, output_dim, params=None):

        params = params or {}

        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=params.get("hidden_units", 128),
            n_layers=params.get("n_layers", 2),
            dropout=params.get("dropout", 0.0)
        )
