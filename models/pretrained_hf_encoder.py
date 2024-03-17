import torch
from torch import nn
from transformers import AutoModel


class PretrainedHFEncoder(nn.Module):
    def __init__(self, model_name, frozen):
        super(PretrainedHFEncoder, self).__init__()

        self.model_name = model_name
        self.frozen = frozen
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(30524)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if not self.frozen:
            output = self.model(input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                output = self.model(input_ids, attention_mask=attention_mask)
        return output
