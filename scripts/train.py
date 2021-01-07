import argparse
import contextlib
import logging
import math
import os
from typing import Generator, Optional

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

    def __init__(self, hyperparameter_lambda: float, ignore_index: int = -100):
        super().__init__(ignore_index=ignore_index)

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


class ImageCaptioningTrainer:
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
        best_models_dir: str = os.path.join(os.environ["TORCH_HOME"], "best_models")
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

        self.best_models_dir = best_models_dir
        if not os.path.exists(self.best_models_dir):
            os.mkdir(self.best_models_dir)

        self.validator = ev.CocoValidator(coco_val_paths, self.coco_train.target_transform.vocabulary)

    def train(
        self,
        *,
        num_epochs: int,
        batch_size: int,
        lr: float,
        loss_lambda: float,
        embedding_dim: int,
        decoder_dim: int,
        attention_dim: int,
        dropout: float,
        maxlen: int,
        patience: int,
        checkpoint_path: Optional[str] = None,
    ) -> float:
        checkpoint = torch.load(checkpoint_path) if checkpoint_path is not None else None

        best_bleu = 0.0
        epochs_without_improvement = 0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on device {torch.cuda.get_device_name(device.index)}")

        with ImageCaptioningTrainer.tensorboard(comment=f"_batch={batch_size}_lr={lr}_lambda={loss_lambda}_dropout={dropout}") as tb:
            data_loader = dl.CocoLoader(self.coco_train, batch_size, math.ceil(os.cpu_count() / 2))

            decoder = model.LSTMDecoder(self.num_embeddings, embedding_dim, self.encoder_dim, decoder_dim, attention_dim, dropout)
            if checkpoint is not None:
                decoder.load_state_dict(checkpoint["decoder"])

            self.encoder.to(device)
            decoder.to(device)

            optimizer = torch.optim.Adam(params=decoder.parameters(), lr=lr)
            if checkpoint is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])

            criterion = DoublyStochasticAttentionLoss(
                hyperparameter_lambda=loss_lambda,
                ignore_index=self.coco_train.target_transform.vocabulary.word2idx("<PAD>"),
            ).to(device)

            # Training loop
            start_epoch = 1 if checkpoint is None else checkpoint["epoch"]
            for epoch in range(start_epoch, start_epoch + num_epochs):
                decoder.train()
                cost = 0.0
                running_loss = 0.0

                # Train steps
                for step, batch in enumerate(data_loader, 1):
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

                    # Print loss every n-th step
                    every_step = 100
                    if not step % every_step:
                        avg_loss = running_loss / every_step
                        running_loss = 0.0
                        logging.info(f"Epoch {epoch} Step {step}/{len(data_loader)} => {avg_loss: .4f}")
                        tb.add_scalar(f"loss_lambda={loss_lambda}", avg_loss, step + (epoch - 1) * len(data_loader))

                self._save_checkpoint(epoch, decoder.state_dict(), optimizer.state_dict(), lr, dropout, loss_lambda)

                current_bleu4 = self.validator.validate(self.encoder, decoder, device)
                logging.info(f"After Epoch {epoch} BLEU-4 => {current_bleu4}")

                self._save_run(epoch, cost / len(data_loader), current_bleu4, loss_lambda, decoder, tb)

                # Early stopping on BLEU-4 metric
                if current_bleu4 > best_bleu:
                    best_bleu = current_bleu4
                    epochs_without_improvement = 0

                    self._save_best_model(embedding_dim, decoder_dim, attention_dim, decoder)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == patience:
                        break

                self.coco_train.shuffle(subset_len=1000)

        return best_bleu

    def _save_checkpoint(self, epoch: int, decoder_state: dict, optim_state: dict, lr: float, dropout: float, loss_lambda: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "decoder": decoder_state,
            "optimizer": optim_state,
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"decoder_lr_{lr}_dropout_{dropout}_lambda_{loss_lambda}.pth")
        torch.save(checkpoint, checkpoint_path)

    def _save_run(self, epoch: int, cost: float, bleu: float, loss_lambda: float, decoder: model.LSTMDecoder, writer: tb.SummaryWriter) -> None:
        writer.add_scalar(f"BLEU-4", bleu, epoch)
        writer.add_scalar(f"cost_lambda={loss_lambda}", cost, epoch)

        for name, weight in decoder.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f"{name}.grad", weight.grad, epoch)

    def _save_best_model(self, embedding_dim: int, decoder_dim: int, attention_dim: int, decoder: model.LSTMDecoder) -> None:
        output_path = os.path.join(self.best_models_dir, f"best_model_e{embedding_dim}_a{attention_dim}_d{decoder_dim}.pth")

        torch.save(decoder.state_dict(), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Captioning with Visual Attention training process")
    parser.add_argument("--num_epochs", default=1, type=int, help="Number of epochs to perform in training process")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate. If checkpoint passed then learning rate will be loaded from state_dict")
    parser.add_argument("--loss_lambda", default=1.0, type=float, help="Value of hyperparameter lambda from loss function")
    parser.add_argument("--embedding_dim", default=128, type=int, help="Word embedding dimmension")
    parser.add_argument("--decoder_dim", default=512, type=int, help="LSTM layer dimmension")
    parser.add_argument("--attention_dim", default=512, type=int, help="Additive attendion dimmension")
    parser.add_argument("--maxlen", default=100, type=int, help="Maximum caption length in words which can be produced during validation step if <EOS> not met")
    parser.add_argument("--patience", default=10, type=int, help="Number of epochs which need to pass while doing early stopping on BLEU-4 metric.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout regularization")
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint path")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d/%m/%Y %H:%M")

    args = parse_args()

    trainer = ImageCaptioningTrainer()
    final_bleu = trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_lambda=args.loss_lambda,
        embedding_dim=args.embedding_dim,
        decoder_dim=args.decoder_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        maxlen=args.maxlen,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
    )

    logging.info(f"Training FINISHED. Final validation BLEU-4 = {final_bleu: .4f}.")
