import lightgbm as lgb
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class ModelFactory:

    @staticmethod
    def criar_lightgbm(task="regression", params=None):
        if task == "regression":
            return lgb.LGBMRegressor(**(params or {}))
        elif task == "classification":
            return lgb.LGBMClassifier(**(params or {}))

    @staticmethod
    def criar_mlp(input_dim, output_dim):
        return MLP(input_dim, output_dim)
