from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class VGG19Encoder(nn.Module):
    "Image Encoder class maps images from pixel to feature space using Oxford VGG19 ConvNet."

    def __init__(self):
        super().__init__()

        self.vgg19 = vgg19(pretrained=True)
        self.vgg19 = self.vgg19.features[:-1]

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, image_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps batch of images from pixel to feature space.

        Args:
            image_batch (torch.Tensor): Tensor of preprocessed images (batch_size, 3, 224, 224).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Flatten feature maps (batch_size, 512, 196).
                                               Feature maps mean (batch_size, 196).
        """
        feature_maps = self.vgg19(image_batch)
        feature_mean = feature_maps.mean(dim=1)

        flatten_dim = 1
        for dim_size in feature_mean.shape[1:]:
            flatten_dim *= dim_size

        feature_maps = feature_maps.view(*feature_maps.shape[0:2], flatten_dim)
        feature_mean = feature_mean.view(feature_mean.shape[0], flatten_dim)

        return feature_maps, feature_mean


class AdditiveAttention(nn.Module):
    def __init__(self, attention_dim: int, values_dim: int, query_dim: int):
        super().__init__()

        self.W_1 = nn.Linear(values_dim, attention_dim)
        self.W_2 = nn.Linear(query_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        values_att = self.W_1(values)
        query_att = self.W_2(query)

        attention_scores = self.v(torch.tanh(values_att + query_att.unsqueeze(1)))
        attention_scores = F.softmax(attention_scores, dim=1)

        context = (values * attention_scores).sum(dim=1)

        return context


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int,
        dropout: float = 0.5
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.init_h = nn.Linear(in_features=encoder_dim, out_features=decoder_dim)
        self.init_c = nn.Linear(in_features=encoder_dim, out_features=decoder_dim)

        self.lstm = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        self.attention = AdditiveAttention(
            attention_dim=attention_dim, values_dim=encoder_dim, query_dim=decoder_dim
        )

        self.dropout = nn.Dropout(p=dropout)

        self.hidden_fc = nn.Linear(in_features=decoder_dim, out_features=embedding_dim)
        self.context_fc = nn.Linear(in_features=encoder_dim, out_features=embedding_dim)
        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=num_embeddings)

    def forward(
        self, feature_maps: torch.Tensor, feature_mean: torch.Tensor, caption_batch
    ) -> torch.Tensor:
        embeddings = self.word_embedding(caption_batch)

        h = self.init_h(feature_mean)
        c = self.init_c(feature_mean)

        predictions = []
        contexts = []

        caption_len = embeddings.shape[1]
        for time_step in range(caption_len - 1):
            embeddings_t = embeddings[:, time_step]

            z = self.attention(feature_maps, h)
            contexts.append(z)

            h, c = self.lstm(torch.cat([embeddings_t, z], dim=1), (h, c))

            out = embeddings_t + self.hidden_fc(self.dropout(h)) + self.context_fc(z)

            preds = self.output_layer(out)
            preds = F.softmax(preds, dim=1)

            predictions.append(preds)

        return torch.stack(predictions), torch.stack(contexts)
