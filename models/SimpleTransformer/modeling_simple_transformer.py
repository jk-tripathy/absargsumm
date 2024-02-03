import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from models.SimpleTransformer import SimpleTransformerConfig


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

    def create_masks(
        self, attention_mask, decoder_attention_mask=None
    ) -> (torch.Tensor, torch.Tensor):
        attention_mask = attention_mask.unsqueeze(1)
        # create subsequent mask
        src_sub_mask = (
            torch.triu(torch.ones(1, attention_mask.size(-1), attention_mask.size(-1))) == 1
        )
        src_mask = attention_mask & src_sub_mask
        # why
        # why must i do this?
        # what does this even mean
        # TODO: pray for salvation
        src_mask = src_mask.repeat(self.config.num_attention_heads, 1, 1)
        src_mask = src_mask.bool().to(attention_mask.device)

        if decoder_attention_mask is not None:
            tgt_attn_mask = decoder_attention_mask.unsqueeze(1)
            # create subsequent mask
            tgt_sub_mask = (
                torch.triu(torch.ones(1, tgt_attn_mask.size(-1), tgt_attn_mask.size(-1))) == 1
            )
            tgt_mask = tgt_attn_mask & tgt_sub_mask
            tgt_mask = tgt_mask.repeat(self.config.num_attention_heads, 1, 1)
            tgt_mask = tgt_mask.bool().to(tgt_attn_mask.device)
        else:
            tgt_mask = None
        return src_mask, tgt_mask

    def shift_tokens_right(
        self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
    ):
        """Shift input ids one token to the right."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_input_embeds=None,
        bos_token_id=69,
        **kwargs,
    ):
        if decoder_input_ids is None and decoder_input_embeds is None:
            decoder_input_ids = self.shift_tokens_right(input_ids, 0, bos_token_id)

        encoder_attentions, decoder_attentions = self.create_masks(
            attention_mask, decoder_attention_mask
        )

        input_embeds = self.encoder_embed(input_ids)
        decoder_input_embeds = self.decoder_embed(decoder_input_ids)
        encoder_last_hidden_state = self.encoder(input_embeds, mask=encoder_attentions)
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
        input_ids,
        attention_mask=None,
        input_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_input_embeds=None,
        bos_token_id=69,
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "input_embeds": input_embeds,
            "decoder_input_embeds": decoder_input_embeds,
            "bos_token_id": bos_token_id,
        }
