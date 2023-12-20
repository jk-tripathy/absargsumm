import pytorch_lightning as pl
import torch
from models.components.base_model import BaseModel
from models.components.gsum_decoder_layer import GSumDecoderLayer
from torch import nn, optim
from transformers import AutoConfig


class GSum(pl.LightningModule):
    def __init__(self, args):
        super(GSum, self).__init__()
        self.args = args

        self.base_model = BaseModel(model_name=args.model_name, frozen=args.frozen)
        self.base_config = AutoConfig.from_pretrained(args.model_name)

        self.source_encoder = nn.TransformerEncoderLayer(
            self.base_config.hidden_size, self.base_config.num_attention_heads
        )
        self.guidance_encoder = nn.TransformerEncoderLayer(
            self.base_config.hidden_size, self.base_config.num_attention_heads
        )

        self.output_decoder = GSumDecoderLayer()

        self.linear = nn.Linear(self.base_config.hidden_size, self.base_config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = nn.NLLLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, target):
        input_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        source_output = self.base_model(input_data)
        source_output = self.source_encoder(source_output)

        guidance_output = self.base_model(input_data)
        guidance_output = self.guidance_encoder(guidance_output)

        output = self.output_decoder(source_output, guidance_output)
        output = self.linear(output)
        output = self.softmax(output)
        output = torch.argmax(output, dim=-1).to(torch.float32)

        loss = self.loss(output.flatten(), target.flatten())
        return output, loss

    def training_step(self, batch, batch_idx):
        output, loss = self.forward(**batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        output, loss = self.forward(**batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        output, loss = self.forward(**batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
