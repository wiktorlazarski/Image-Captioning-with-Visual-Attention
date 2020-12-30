import contextlib
import logging as log
import math
import os
from typing import Generator

import torch
import torch.nn as nn
from torch.utils import tensorboard as tb
from torchvision import transforms

import scripts.data_loading as dl
import scripts.data_processing as dp
import scripts.eval as ev
from scripts import model


class DoublyStochasticAttentionLoss(nn.CrossEntropyLoss):
    """Computes loss function for double stochastic attention model."""

    def __init__(self, hyperparameter_lambda: float):
        super().__init__()

        self.hyperparameter_lambda = hyperparameter_lambda

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor, alphas: torch.tensor) -> float:
        """Computes loss function for double stochastic attention model.

        Args:
            y_pred (torch.tensor): predictions for every time step (time_step * batch_size, vocabulary_size)
            y_true (torch.tensor): true prediction for every time step (time_step * batch_size)
            alphas (torch.tensor): attention scores produced for every prediction (time_step, batch_size, num_feature_maps)

        Returns:
            float: loss value
        """
        loss = super().forward(y_pred, y_true)

        loss += self.hyperparameter_lambda * ((1.0 - alphas.sum(dim=0)) ** 2).sum(dim=1).mean()

        return loss


class Trainer:
    @staticmethod
    @contextlib.contextmanager
    def tensorboard(comment: str) -> Generator[tb.SummaryWriter, None, None]:
        writer = tb.SummaryWriter(comment=comment)
        try:
            yield writer
        finally:
            writer.close()

    def __init__(
        self,
        coco_train_paths: dl.CocoTrainingDatasetPaths = dl.DATASET_PATHS[dl.DatasetType.TRAIN],
        coco_val_paths: dl.CocoTrainingDatasetPaths = dl.DATASET_PATHS[dl.DatasetType.VALIDATION],
        image_pipeline: transforms.transforms = dp.VGGNET_PREPROCESSING_PIPELINE,
        caption_pipeline: dp.TextPipeline = dp.TextPipeline(),
        checkpoint_dir: str = os.path.join(os.environ["TORCH_HOME"], "checkpoints"),
    ):
        self.coco_train = dl.CocoCaptions(
            dset_paths=coco_train_paths,
            transform=image_pipeline,
            target_transform=caption_pipeline,
        )

        self.num_embeddings = len(self.coco_train.target_transform.vocabulary)
        self.encoder_dim = 196

        self.encoder = model.VGG19Encoder()

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.validator = ev.CocoValidator(coco_val_paths, self.coco_train.target_transform.vocabulary)

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        loss_lambda: float,
        embedding_dim: int,
        decoder_dim: int,
        attention_dim: int,
        dropout: float,
    ) -> None:
        comment = f"_batch={batch_size}_lr={learning_rate}_lambda={loss_lambda}_dropout={dropout}"

        with Trainer.tensorboard(comment=comment) as tb:
            data_loader = dl.CocoLoader(
                self.coco_train, batch_size=batch_size, num_workers=math.ceil(os.cpu_count() / 2)
            )

            decoder = model.LSTMDecoder(
                num_embeddings=self.num_embeddings,
                embedding_dim=embedding_dim,
                encoder_dim=self.encoder_dim,
                decoder_dim=decoder_dim,
                attention_dim=attention_dim,
                dropout=dropout,
            )

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            log.info(f"Training on device {torch.cuda.get_device_name(device.index)}")

            self.encoder.to(device)
            decoder.to(device)
            decoder.train()

            optimizer = torch.optim.Adam(params=decoder.parameters(), lr=learning_rate)
            criterion = DoublyStochasticAttentionLoss(loss_lambda).to(device)

            for epoch in range(num_epochs):
                cost = 0.0
                running_loss = 0.0

                for step, batch in enumerate(data_loader):
                    images, captions = batch[0].to(device), batch[1].to(device)

                    optimizer.zero_grad()

                    predictions, attentions = decoder(*self.encoder(images), captions)

                    num_samples, caption_len = captions.shape[0], captions.shape[1] - 1

                    loss = criterion(
                        predictions.reshape(caption_len * num_samples, self.num_embeddings),
                        captions[:, 1:].reshape(caption_len * num_samples),
                        attentions,
                    )

                    loss.backward()
                    optimizer.step()

                    cost += loss.item()
                    running_loss += loss.item()

                    every_step = 10
                    if step % every_step == 0 and step != 0:
                        avg_loss = running_loss / every_step
                        running_loss = 0.0
                        log.info(f"Epoch {epoch + 1} Step {step}/{len(data_loader)} => {avg_loss}")

                self._save_checkpoint(
                    epoch=epoch,
                    decoder_state=decoder.state_dict(),
                    optim_state=optimizer.state_dict(),
                    lr=learning_rate,
                    dropout=dropout,
                    loss_lambda=loss_lambda,
                )

                bleu = self.validator.validate(self.encoder, decoder, device)
                log.info(f"Epoch {epoch + 1} BLEU => {bleu}")

                self._save_tensorboard_data(
                    epoch=epoch,
                    cost=cost / len(data_loader),
                    bleu=bleu,
                    loss_lambda=loss_lambda,
                    decoder=decoder,
                    writer=tb,
                )

                self.coco_train.shuffle(subset_len=500)

    def _save_checkpoint(
        self,
        epoch: int,
        decoder_state: dict,
        optim_state: dict,
        lr: float,
        dropout: float,
        loss_lambda: float,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "decoder": decoder_state,
            "optimizer": optim_state,
        }

        checkpoint_output = os.path.join(
            self.checkpoint_dir,
            f"model_lr_{lr}_dropout_{dropout}_lambda_{loss_lambda}.pth",
        )
        torch.save(checkpoint, checkpoint_output)

    def _save_tensorboard_data(
        self,
        epoch: int,
        cost: float,
        bleu: float,
        loss_lambda: float,
        decoder: model.LSTMDecoder,
        writer: tb.SummaryWriter,
    ) -> None:
        writer.add_scalar(f"BLEU-4", bleu, epoch)
        writer.add_scalar(f"cost_lambda={loss_lambda}", cost, epoch)

        for name, weight in decoder.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f"{name}.grad", weight.grad, epoch)


if __name__ == "__main__":
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M",
    )

    trainer = Trainer()
    trainer.train(
        num_epochs=2,
        batch_size=16,
        learning_rate=0.00005,
        loss_lambda=0.001,
        embedding_dim=128,
        decoder_dim=256,
        attention_dim=256,
        dropout=0.2,
    )
