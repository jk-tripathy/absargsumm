import pytorch_lightning as pl
from torch import nn, optim
from transformers import AutoConfig

from models.components.base_model import BaseModel
from models.components.gsum_decoder_layer import GSumDecoderLayer


class GSum(pl.LightningModule):
    def __init__(self, args):
        super(GSum, self).__init__()
        self.args = args

        self.base_model = BaseModel(model_name=args.model_name, frozen=args.frozen)
        self.base_config = AutoConfig.from_pretrained(args.model_name)

        self.source_transformer = nn.TransformerEncoderLayer(
            self.base_config.hidden_size, self.base_config.num_attention_heads
        )
        self.guidance_transformer = nn.TransformerEncoderLayer(
            self.base_config.hidden_size, self.base_config.num_attention_heads
        )

        self.output_transformer = GSumDecoderLayer()

        self.linear = nn.Linear(self.base_config.hidden_size, self.base_config.vocab_size)
        self.softmax = nn.Softmax()

        self.loss = nn.NLLLoss(ignore_index=0, reduction="sum")

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        input_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        source_output = self.base_model(input_data)
        source_output = self.source_transformer(source_output)

        guidance_output = self.base_model(input_data)
        guidance_output = self.guidance_transformer(guidance_output)

        output = self.output_transformer(source_output, guidance_output)
        output = self.linear(output)
        output = self.softmax(output)

        loss = self.loss(output, labels)
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
