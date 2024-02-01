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

        self.loss = torch.nn.CrossEntropyLoss()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(1, sz, sz)) == 1
        mask = (
            mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        )
        return mask.bool()

    def create_masks(self, src_attn_mask, tgt_attn_mask=None):
        src_attn_mask = src_attn_mask.unsqueeze(1)
        src_sub_mask = self.generate_square_subsequent_mask(src_attn_mask.size(-1))
        src_mask = src_attn_mask & src_sub_mask
        # why
        # why must i do this?
        # what does this even mean
        # TODO: pray for salvation
        src_mask = src_mask.repeat(self.base_config.num_attention_heads, 1, 1)
        src_mask = src_mask.bool().to(src_attn_mask.device)

        if tgt_attn_mask is not None:
            tgt_attn_mask = tgt_attn_mask.unsqueeze(1)
            tgt_sub_mask = self.generate_square_subsequent_mask(tgt_attn_mask.size(-1))
            tgt_mask = tgt_attn_mask & tgt_sub_mask
            tgt_mask = tgt_mask.repeat(self.base_config.num_attention_heads, 1, 1)
            tgt_mask = tgt_mask.bool().to(tgt_attn_mask.device)
        return src_mask, tgt_mask

    def forward(self, src, src_attn_mask, tgt, tgt_attn_mask):
        src_mask, tgt_mask = self.create_masks(src_attn_mask, tgt_attn_mask)

        src_embed_out = self.encoder_embed(src)
        tgt_embed_out = self.decoder_embed(tgt)
        enc_out = self.encoder(src_embed_out, mask=src_mask)
        dec_out_logits = self.decoder(
            tgt_embed_out, enc_out, tgt_mask=tgt_mask, memory_mask=src_mask
        )
        dec_out_logits = self.linear(dec_out_logits)

        loss = self.loss(dec_out_logits.permute(0, 2, 1), tgt)

        return dec_out_logits, loss

    def training_step(self, batch, batch_idx):
        output, loss = self.forward(**batch)
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
