from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGG19Encoder(nn.Module):
    "Image Encoder class maps images from pixel to feature space using Oxford VGG19 ConvNet."

    def __init__(self):
        super().__init__()

        self.vgg19 = vgg19(pretrained=True)
        self.vgg19 = self.vgg19.features[:-1]

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, image_batch: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Maps batch of images from pixel to feature space.

        Args:
            image_batch (torch.tensor): tensor of preprocessed images (batch_size, 3, 224, 224).

        Returns:
            Tuple[torch.tensor, torch.tensor]: Flatten feature maps (batch_size, 512, 196).
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
    """Computes Additive/Bahdanau attention."""

    def __init__(self, attention_dim: int, values_dim: int, query_dim: int):
        super().__init__()

        self.W_1 = nn.Linear(values_dim, attention_dim)
        self.W_2 = nn.Linear(query_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(
        self, values: torch.tensor, query: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Output context vector.

        Args:
            values (torch.tensor): Flatten feature maps (batch_size, num_feature_maps, feature_map_dim).
            query (torch.tensor): LSTM decoder output (batch_size, decoder_dim)

        Returns:
            torch.tensor: Context vector (batch_size, feature_map_dim)
            torch.tensor: Attention scores (batch_size, num_feature_maps)
        """
        values_att = self.W_1(values)
        query_att = self.W_2(query)

        attention_scores = self.v(torch.tanh(values_att + query_att.unsqueeze(1)))
        attention_scores = torch.softmax(attention_scores, dim=1)

        context = (values * attention_scores).sum(dim=1)

        return context, attention_scores.squeeze(-1)


class LSTMDecoder(nn.Module):
    """Long Short Term Memory decoder with Additive Attention."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int,
        dropout: float = 0.5,
    ):
        """Constructor

        Args:
            num_embeddings (int): vocabulary size
            embedding_dim (int): word embedding dimension
            encoder_dim (int): encoder output of flatten feature mean dimension
            decoder_dim (int): decoder output dimension
            attention_dim (int): additive attention internal dimension
            dropout (float, optional): decoder output dropout. Defaults to 0.5.
        """
        super().__init__()

        self.word_embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.init_h = nn.Linear(in_features=encoder_dim, out_features=decoder_dim)
        self.init_c = nn.Linear(in_features=encoder_dim, out_features=decoder_dim)

        self.lstm = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        self.attention = AdditiveAttention(
            attention_dim=attention_dim, values_dim=encoder_dim, query_dim=decoder_dim
        )

        self.beta_fc = nn.Linear(in_features=decoder_dim, out_features=1)

        self.dropout = nn.Dropout(p=dropout)

        self.hidden_fc = nn.Linear(in_features=decoder_dim, out_features=embedding_dim)
        self.context_fc = nn.Linear(in_features=encoder_dim, out_features=embedding_dim)
        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=num_embeddings)

    def forward(
        self, feature_maps: torch.tensor, feature_mean: torch.tensor, caption_batch: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Decoder forward pass.

        Args:
            feature_maps (torch.tensor): Flatten feature maps (batch_size, num_feature_maps, feature_map_dim).
            feature_mean (torch.tensor): Flatten feature maps mean (batch_size, feature_map_dim).
            caption_batch ([type]): Image captions (batch_size, caption_len, token_index)

        Returns:
            Tuple[torch.tensor, torch.tensor]: Prediction at each time step (time_step, batch_size, vocabulary_size)
                                               Context vectors for each prediction (time_step, batch_size, encoder_dim)

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: Prediction at each time step (time_step, batch_size, vocabulary_size)
                                                             Context vectors for each prediction (time_step, batch_size, encoder_dim)
                                                             Attention scores (time_step, batch_size, num_feature_maps)
        """
        embeddings = self.word_embedding(caption_batch)

        h = self.init_h(feature_mean)
        c = self.init_c(feature_mean)

        predictions = []
        contexts = []
        attention_scores = []

        caption_len = embeddings.shape[1]
        for time_step in range(caption_len - 1):
            embeddings_t = embeddings[:, time_step]

            z, alphas = self.attention(feature_maps, h)

            beta = torch.sigmoid(self.beta_fc(h))
            z = z * beta

            contexts.append(z)
            attention_scores.append(alphas)

            h, c = self.lstm(torch.cat([embeddings_t, z], dim=1), (h, c))

            out = embeddings_t + self.hidden_fc(self.dropout(h)) + self.context_fc(z)

            preds = self.output_layer(out)

            predictions.append(preds)

        return torch.stack(predictions), torch.stack(contexts), torch.stack(attention_scores)
