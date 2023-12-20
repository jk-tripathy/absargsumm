import torch.nn as nn


class GSumDecoderLayer(nn.Module):
    def __init__(self):
        super(GSumDecoderLayer, self).__init__()
        self.dummy_linear = nn.Linear(768, 768)

    def forward(self, x):
        return self.dummy_linear(x)
