import pytorch_lightning as pl
import torch
from torch import nn, optim
from transformers import AutoConfig

from models.base_model import BaseModel
from models.GSum.gsum_decoder import GSumDecoder
from models.GSum.gsum_decoder_layer import GSumDecoderLayer


class GSum(pl.LightningModule):
    def __init__(self, args):
        super(GSum, self).__init__()
        self.args = args

        self.base_model = BaseModel(model_name=args.model_name, frozen=args.frozen)
        self.base_config = AutoConfig.from_pretrained(args.model_name)

        self.target_embed = nn.Embedding(self.base_config.vocab_size, self.base_config.hidden_size)

        self.source_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.base_config.hidden_size,
            nhead=self.base_config.num_attention_heads,
            batch_first=args.batch_first,
        )
        self.guidance_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.base_config.hidden_size,
            nhead=self.base_config.num_attention_heads,
            batch_first=args.batch_first,
        )

        self.output_decoder_layer = GSumDecoderLayer(
            d_model=self.base_config.hidden_size,
            nhead=self.base_config.num_attention_heads,
            batch_first=args.batch_first,
        )
        self.output_decoder = GSumDecoder(self.output_decoder_layer, num_layers=12)

        self.linear = nn.Linear(self.base_config.hidden_size, self.base_config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = nn.NLLLoss()

    def _source_encoder(self, x):
        x = self.base_model(x)
        x = self.source_transformer_layer(x)
        return x

    def _guidance_encoder(self, x):
        x = self.base_model(x)
        x = self.guidance_transformer_layer(x)
        return x

    def forward(self, inputs, targets):
        targt_embeded = self.target_embed(targets)
        source_output = self._source_encoder(inputs)
        guidance_output = self._guidance_encoder(inputs)
        decoder_output = self.output_decoder(
            source=source_output,
            source_mask=None,
            guidance=guidance_output,
            guidance_mask=None,
            target=targt_embeded,
            target_mask=None,
        )

        x = self.linear(decoder_output)
        x = self.softmax(x)

        output = torch.argmax(x, dim=-1).to(torch.float32)

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
