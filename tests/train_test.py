import torch
from scripts import train


def test_cuda_availability() -> None:
    assert torch.cuda.is_available()


def test_doubly_stochastic_attention_loss() -> None:
    # given
    # - ln(1/2) = 0.6931
    y_pred = torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    y_true = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    caption_len = 2
    batch_size = 2

    attention_scores = torch.tensor(
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]], dtype=torch.float
    )

    loss = train.DoublyStochasticAttentionLoss(hyperparameter_lambda=1.0)

    expected_result = 3.6931

    # when
    result = loss.forward(
        y_pred.reshape(caption_len * batch_size, - 1),
        y_true.reshape(caption_len * batch_size),
        attention_scores
    )

    # then
    assert round(result.item(), 4) == expected_result
