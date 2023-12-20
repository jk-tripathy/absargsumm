import torch
from torch import nn
from transformers import AutoModel


class BaseModel(nn.Module):
    def __init__(self, model_name, frozen=True):
        super(BaseModel, self).__init__()

        self.model_name = model_name
        self.frozen = frozen
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, x):
        if not self.frozen:
            logits = self.model(**x).logits
        else:
            with torch.no_grad():
                logits = self.model(**x).logits
        return logits
