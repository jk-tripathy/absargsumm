import pytorch_lightning as pl
import torch
from torch import nn, optim
from transformers import AutoConfig

from models.base_model import BaseModel
from models.GSum.gsum_decoder_layer import GSumDecoderLayer


class GSum(pl.LightningModule):
    def __init__(self, args):
        super(GSum, self).__init__()
        self.args = args

        self.base_model = BaseModel(model_name=args.model_name, frozen=args.frozen)
        self.base_config = AutoConfig.from_pretrained(args.model_name)

        self.source_transformer_layer = nn.TransformerEncoderLayer(
            self.base_config.hidden_size, self.base_config.num_attention_heads, batch_first=True
        )
        self.guidance_transformer_layer = nn.TransformerEncoderLayer(
            self.base_config.hidden_size, self.base_config.num_attention_heads, batch_first=True
        )

        self.output_decoder = GSumDecoderLayer()

        self.linear = nn.Linear(self.base_config.hidden_size, self.base_config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = nn.NLLLoss()

    def _source_encoder(self, input_data):
        source_output = self.base_model(input_data)
        source_output = self.source_transformer_layer(source_output)
        return source_output

    def _guidance_encoder(self, input_data):
        guidance_output = self.base_model(input_data)
        guidance_output = self.guidance_transformer_layer(guidance_output)
        return guidance_output

    def forward(self, inputs, targets):
        source_output = self._source_encoder(inputs)
        guidance_output = self._guidance_encoder(inputs)

        output = self.output_decoder(source_output, guidance_output)
        output = self.linear(output)
        output = self.softmax(output)
        output = torch.argmax(output, dim=-1).to(torch.float32)

        loss = self.loss(output.flatten(), targets.flatten())
        return output, loss

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        output, loss = self.forward(inputs, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        output, loss = self.forward(inputs, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        output, loss = self.forward(inputs, targets)
        self.log("test_loss", loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
