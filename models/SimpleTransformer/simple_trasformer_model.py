import pytorch_lightning as pl
import torch
from transformers import AutoConfig


class SimpleTransformer(pl.LightningModule):
    def __init__(self, args):
        super(SimpleTransformer, self).__init__()
        self.args = args
        self.base_config = AutoConfig.from_pretrained(args.model_name)

        self.encoder_embed = torch.nn.Embedding(
            self.base_config.vocab_size, self.base_config.hidden_size
        )
        self.decoder_embed = torch.nn.Embedding(
            self.base_config.vocab_size, self.base_config.hidden_size
        )

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.base_config.hidden_size,
            nhead=self.base_config.num_attention_heads,
            batch_first=args.batch_first,
        )
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=12)

        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=self.base_config.hidden_size,
            nhead=self.base_config.num_attention_heads,
            batch_first=args.batch_first,
        )
        self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=12)

        self.linear = torch.nn.Linear(self.base_config.hidden_size, self.base_config.vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.loss = torch.nn.NLLLoss()

    def forward(self, inputs, targets):
        inputs = self.encoder_embed(inputs["input_ids"])
        outputs = self.decoder_embed(targets)

        inputs = self.encoder(inputs)
        outputs = self.decoder(outputs, inputs)

        outputs = self.linear(outputs)
        outputs = self.softmax(outputs)

        outputs = torch.argmax(outputs, dim=-1).to(torch.float32)

        loss = self.loss(outputs.flatten(), targets.flatten())

        return outputs, loss

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
