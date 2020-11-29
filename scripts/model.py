from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGG19Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg19 = vgg19(pretrained=True)
        self.vgg19 = self.vgg19.features[:-1]

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.vgg19(image)[0]
        feature_mean = feature_maps.mean(dim=0)

        flatten_dim = 1
        for dim_size in feature_mean.shape:
            flatten_dim *= dim_size

        feature_maps = feature_maps.view(feature_maps.shape[0], flatten_dim)
        feature_mean = feature_mean.view(flatten_dim)

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
