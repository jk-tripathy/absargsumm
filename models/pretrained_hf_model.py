import torch
from torch import nn
from transformers import AutoModel


class PretrainedHFEncoder(nn.Module):
    def __init__(self, model_name, frozen=True):
        super(PretrainedHFEncoder, self).__init__()

        self.model_name = model_name
        self.frozen = frozen
        self.model = AutoModel.from_pretrained(model_name)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, **kwargs):
        if not self.frozen:
            logits = self.model(**x).last_hidden_state
        else:
            with torch.no_grad():
                logits = self.model(**x).last_hidden_state
        return logits
