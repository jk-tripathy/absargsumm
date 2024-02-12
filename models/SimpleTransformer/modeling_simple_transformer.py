from typing import Dict, Optional

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from models.SimpleTransformer import SimpleTransformerConfig
from utils import create_masks, shift_tokens_right


class SimpleTransformer(PreTrainedModel):
    config_class = SimpleTransformerConfig

    def __init__(self, config: SimpleTransformerConfig):
        super(SimpleTransformer, self).__init__(config)
        self.config = config
        self.encoder_embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            batch_first=config.batch_first,
        )
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=12)

        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            batch_first=config.batch_first,
        )
        self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=12)

        self.linear = torch.nn.Linear(config.hidden_size, config.vocab_size)

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_embeds: Optional[torch.tensor] = None,
        decoder_input_ids: Optional[torch.tensor] = None,
        decoder_attention_mask: Optional[torch.tensor] = None,
        decoder_input_embeds: Optional[torch.tensor] = None,
        bos_token_id=69,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if decoder_input_ids is None and decoder_input_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, 0, bos_token_id)

        encoder_attentions = create_masks(
            attention_mask,
            expand_dims=True,
            num_attention_heads=self.config.num_attention_heads,
            for_causal=False,
        )
        if decoder_attention_mask is not None:
            decoder_attentions = create_masks(
                decoder_attention_mask,
                expand_dims=True,
                num_attention_heads=self.config.num_attention_heads,
                for_causal=True,
            )
        else:
            decoder_attentions = None

        input_embeds = self.encoder_embed(input_ids)
        decoder_input_embeds = self.decoder_embed(decoder_input_ids)
        encoder_last_hidden_state = self.encoder(
            input_embeds,
            mask=encoder_attentions,
        )
        decoder_last_hidden_state = self.decoder(
            decoder_input_embeds,
            encoder_last_hidden_state,
            tgt_mask=decoder_attentions,
            memory_mask=encoder_attentions,
        )
        lm_logits = self.linear(decoder_last_hidden_state)

        loss = self.loss(lm_logits.permute(0, 2, 1), decoder_input_ids)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            decoder_attentions=decoder_attentions,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_attentions=encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        bos_token_id: int = 69,
        **kwargs,
    ) -> Dict[str, torch.tensor]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bos_token_id": bos_token_id,
        }
