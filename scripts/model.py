from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import vgg19


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.vgg19 = vgg19(pretrained=True)
        self.vgg19 = self.vgg19.features[:-1]

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, image_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.vgg19(image_batch)
        # encode in order to pass to RNN h, c
        initial_rnn = None
        return feature_maps, initial_rnn
