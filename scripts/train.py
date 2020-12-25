import torch
import torch.nn as nn

from scripts import model


class DoublyStochasticAttentionLoss(nn.CrossEntropyLoss):
    """Computes loss function for double stochastic attention model."""

    def __init__(self, hyperparameter_lambda: float):
        super().__init__()

        self.hyperparameter_lambda = hyperparameter_lambda

    def forward(self, y_preds: torch.tensor, y_true: torch.tensor, alphas: torch.tensor) -> float:
        """Computes loss function for double stochastic attention model.

        Args:
            y_preds (torch.tensor): predictions for every time step (time_step, batch_size, vocabulary_size)
            y_true (torch.tensor): true prediction for every time step (time_step, batch_size)
            alphas (torch.tensor): attention scores produced for every prediction (time_step, batch_size, num_feature_maps)

        Returns:
            float: loss value
        """
        loss = super().forward(y_preds, y_true)

        loss += self.hyperparameter_lambda * ((1 - alphas.sum(dim=0)) ** 2).sum(dim=1).mean()

        return loss


class Trainer:
    def __init__(self, parameters):
        pass

    def train() -> None:
        pass
