from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from transformers import PreTrainedModel
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import ModelOutput

from models.GSum import GSumConfig
from models.pretrained_hf_encoder import PretrainedHFEncoder
from utils import create_masks, shift_tokens_right


@dataclass
class GSumEncoderBaseModelOutput(ModelOutput):
    source_last_hidden_state: torch.tensor = None
    guidance_last_hidden_state: torch.tensor = None
    source_3D_attentions: torch.tensor = None
    guidance_3D_attentions: torch.tensor = None


@dataclass
class GSumSeq2SeqLMOutput(ModelOutput):
    loss: torch.tensor = None
    logits: torch.tensor = None
    source_last_hidden_state: torch.Tensor = None
    guidance_last_hidden_state: torch.Tensor = None
    decoder_last_hidden_state: torch.tensor = None


# copied from transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding
class GSumLearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class GSumEncoder(nn.Module):
    def __init__(self, config: GSumConfig):
        super(GSumEncoder, self).__init__()
        self.config = config
        self.pretrained_hf_encoder = PretrainedHFEncoder(
            model_name=self.config.pretrained_encoder_name_or_path,
            frozen=self.config.freeze_encoder,
        )

        self.source_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            batch_first=self.config.batch_first,
        )

        self.guidance_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            batch_first=self.config.batch_first,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        guidance_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> GSumEncoderBaseModelOutput:
        source_attentions = _prepare_4d_attention_mask(
            mask=attention_mask,
            dtype=torch.float32,
        )
        guidance_attentions = _prepare_4d_attention_mask(
            mask=guidance_attention_mask,
            dtype=torch.float32,
        )

        source_encoder_output = self.pretrained_hf_encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        source_logits = self.source_transformer_layer(
            source_encoder_output.last_hidden_state,
        )
        guidance_encoder_output = self.pretrained_hf_encoder(
            guidance_input_ids,
            attention_mask=guidance_attention_mask,
        )
        guidance_logits = self.guidance_transformer_layer(
            guidance_encoder_output.last_hidden_state,
        )
        return GSumEncoderBaseModelOutput(
            source_last_hidden_state=source_logits,
            guidance_last_hidden_state=guidance_logits,
            source_3D_attentions=source_attentions,
            guidance_3D_attentions=guidance_attentions,
        )


class GSumDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GSumConfig,
    ):
        super(GSumDecoderLayer, self).__init__()

        self.config = config

        self.self_attn = MultiheadAttention(
            self.config.d_model,
            self.config.nhead,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.guidance_multihead_attn = MultiheadAttention(
            self.config.d_model,
            self.config.nhead,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.source_multihead_attn = MultiheadAttention(
            self.config.d_model,
            self.config.nhead,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )

        self.dropout = Dropout(self.config.dropout)

        self.norm1 = LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.norm2 = LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.norm3 = LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.norm4 = LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)

        # Implementation of Feedforward model
        self.linear1 = Linear(self.config.d_model, self.config.decoder_ff_dim)
        self.linear2 = Linear(self.config.decoder_ff_dim, self.config.d_model)

        self.activation = F.relu

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm4(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward(
        self,
        source_logits: torch.Tensor,
        guidance_logits: torch.Tensor,
        target_embeds: torch.Tensor,
        source_attentions: Optional[torch.Tensor] = None,
        guidance_attentions: Optional[torch.Tensor] = None,
        target_attentions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.norm1(target_embeds)
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=target_attentions)
        x = self.dropout(self_attn_out) + x
        x = self.norm2(x)

        guidance_multihead_attn_out, _ = self.guidance_multihead_attn(
            guidance_logits, guidance_logits, x, attn_mask=guidance_attentions
        )
        x = self.dropout(guidance_multihead_attn_out) + x
        x = self.norm3(x)

        source_multihead_attn_out, _ = self.source_multihead_attn(
            source_logits, source_logits, x, attn_mask=source_attentions
        )
        x = self.dropout(source_multihead_attn_out) + x

        out = self._ff_block(x)
        return out


class GSumDecoder(nn.Module):
    def __init__(
        self,
        config: GSumConfig,
    ):
        super(GSumDecoder, self).__init__()
        self.config = config
        self.config.is_decoder = True

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.embed_positions = GSumLearnedPositionalEmbedding(
            self.config.max_position_embeddings,
            self.config.d_model,
        )
        self.decoder_layer = GSumDecoderLayer(self.config)

        self.layers = nn.ModuleList(
            [self.decoder_layer for _ in range(self.config.num_decoder_layers)]
        )
        self.norm = LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)

    def forward(
        self,
        source_logits: torch.Tensor,
        guidance_logits: torch.Tensor,
        target_input_ids: torch.Tensor,
        source_attentions: Optional[torch.Tensor] = None,
        guidance_attentions: Optional[torch.Tensor] = None,
        target_attentions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        source_attentions = _prepare_4d_attention_mask(
            mask=source_attentions,
            dtype=torch.float32,
        )
        guidance_attentions = _prepare_4d_attention_mask(
            mask=guidance_attentions,
            dtype=torch.float32,
        )

        if target_attentions is not None:
            target_attentions = _create_4d_causal_attention_mask(
                input_shape=target_attentions.shape,
                dtype=torch.float32,
                device=target_attentions.device,
            )

        target_embeds = self.embedding(target_input_ids) + self.embed_positions(target_input_ids)
        x = None
        for layer in self.layers:
            x = layer(
                source_logits=source_logits,
                source_3D_attentions=source_attentions,
                guidance_logits=guidance_logits,
                guidance_3D_attentions=guidance_attentions,
                target_embeds=target_embeds,
                target_3D_attentions=target_attentions,
            )

        x = self.norm(x)
        return x


class GSum(PreTrainedModel):
    def __init__(self, config: GSumConfig):
        super(GSum, self).__init__(config)
        self.config = config

        self.encoder = GSumEncoder(self.config)
        self.decoder = GSumDecoder(self.config)

        self.linear = nn.Linear(self.config.d_model, self.config.vocab_size)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="sum")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        guidance_input_ids: torch.Tensor = None,
        guidance_attention_mask: torch.Tensor = None,
        input_embeds: Optional[torch.tensor] = None,
        guidance_input_embeds: Optional[torch.tensor] = None,
        decoder_input_ids: Optional[torch.tensor] = None,
        decoder_attention_mask: Optional[torch.tensor] = None,
        decoder_input_embeds: Optional[torch.tensor] = None,
        **kwargs,
    ):
        if decoder_input_ids is None and decoder_input_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, 0, self.config.bos_token_id)

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            guidance_input_ids=guidance_input_ids,
            guidance_attention_mask=guidance_attention_mask,
        )

        decoder_last_hidden_state = self.decoder(
            source_logits=encoder_output.source_last_hidden_state,
            source_attentions=attention_mask,
            guidance_logits=encoder_output.guidance_last_hidden_state,
            guidance_attentions=guidance_attention_mask,
            target_input_ids=decoder_input_ids,
            target_attentions=decoder_attention_mask,
        )
        print(f"decoder_last_hidden_state: {decoder_last_hidden_state}")

        lm_logits = self.linear(decoder_last_hidden_state)
        print(f"lm_logits: {lm_logits}")

        loss = self.loss(lm_logits.view(-1, self.config.vocab_size), decoder_input_ids.view(-1))

        return GSumSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            source_last_hidden_state=encoder_output.source_last_hidden_state,
            guidance_last_hidden_state=encoder_output.guidance_last_hidden_state,
            decoder_last_hidden_state=decoder_last_hidden_state,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        guidance_input_ids: torch.Tensor,
        guidance_attention_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.tensor]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "guidance_input_ids": guidance_input_ids,
            "guidance_attention_mask": guidance_attention_mask,
        }
