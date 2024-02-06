from typing import Dict, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(
    args: Dict[str, Union[str, int, bool]],
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    special_tokens_dict = {
        "bos_token": args.bos_token,
        "eos_token": args.eos_token,
    }
    tokenizer.add_special_tokens(
        special_tokens_dict,
        replace_additional_special_tokens=True,
    )

    return tokenizer


def shift_tokens_right(
    input_ids: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    """Shift input ids one token to the right."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # replace any eos token in the input with pad token
    shifted_input_ids.masked_fill_(shifted_input_ids == decoder_start_token_id, pad_token_id)

    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def create_masks(
    num_attention_heads: int,
    attention_mask: torch.Tensor,
    decoder_attention_mask: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
    attention_mask = attention_mask.unsqueeze(1)
    # create subsequent mask
    src_sub_mask = torch.triu(torch.ones(1, attention_mask.size(-1), attention_mask.size(-1))) == 1
    src_mask = attention_mask & src_sub_mask
    # TODO:
    # src_mask needs to be bsz*nhead for multiheadattention but just bsz for BERT
    # wut do?

    src_mask = src_mask.repeat(num_attention_heads, 1, 1)
    src_mask = src_mask.bool().to(attention_mask.device)

    if decoder_attention_mask is not None:
        tgt_attn_mask = decoder_attention_mask.unsqueeze(1)
        # create subsequent mask
        tgt_sub_mask = (
            torch.triu(torch.ones(1, tgt_attn_mask.size(-1), tgt_attn_mask.size(-1))) == 1
        )
        tgt_mask = tgt_attn_mask & tgt_sub_mask
        tgt_mask = tgt_mask.repeat(num_attention_heads, 1, 1)
        tgt_mask = tgt_mask.bool().to(tgt_attn_mask.device)
    else:
        tgt_mask = None
    return src_mask, tgt_mask
