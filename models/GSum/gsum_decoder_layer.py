import torch
import torch.nn as nn


class GSumDecoderLayer(nn.Module):
    def __init__(self):
        super(GSumDecoderLayer, self).__init__()

    def forward(self, x, y):
        return torch.randn(x.shape[0], x.shape[1], x.shape[2])
