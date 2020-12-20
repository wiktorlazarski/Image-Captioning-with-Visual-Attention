import torch
import torch.nn as nn


class DoublyStochasticAttentionLoss(nn.CrossEntropyLoss):
    def __init__(self, hyperparameter_lambda: float):
        super().__init__()

        self.hyperparameter_lambda = hyperparameter_lambda

    def forward(self, y_preds: torch.tensor, y_true: torch.tensor, alphas: torch.tensor) -> float:
        loss = super().forward(y_preds, y_true)

        loss += self.hyperparameter_lambda * ((1 - alphas.sum(dim=0)) ** 2).sum(dim=1).mean()

        return loss
