from base_model import BaseModel
from components.gsum_decoder_layer import GSumDecoderLayer
from torch import nn


class GSum(nn.Module):
    def __init__(self, model_name):
        super(GSum, self).__init__()
        self.base_model = BaseModel(model_name)

        self.source_transformer = nn.TransformerEncoderLayer()
        self.guidance_transformer = nn.TransformerEncoderLayer()

        self.output_transformer = GSumDecoderLayer()

        self.linear = nn.Linear()
        self.softmax = nn.Softmax()

    def forward(self, x):
        source_output = self.base_model(x)
        source_output = self.source_transformer(source_output)

        guidance_output = self.base_model(x)
        guidance_output = self.guidance_transformer(guidance_output)

        output = self.output_transformer(source_output, guidance_output)
        output = self.linear(output)
        output = self.softmax(output)
        return output
