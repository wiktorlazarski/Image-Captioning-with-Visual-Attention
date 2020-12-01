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


class VisualAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass


class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass
