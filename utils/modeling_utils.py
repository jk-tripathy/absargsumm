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
    attention_mask: torch.Tensor,
    expand_dims: bool = False,
    for_causal: bool = False,
    num_attention_heads: int = None,
    decoder_attention_mask: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
    # our mask is used to prevent the model from attending to the padding tokens
    # attention mask returned by the tokenizer is 0 for padding tokens and 1 for non-padding tokens
    # so we need to invert it
    # READ: https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
    inverted_mask = attention_mask == 0

    mask = inverted_mask.unsqueeze(1)
    mask = mask.repeat(1, mask.size(-1), 1)

    # create subsequent mask
    if for_causal:
        sub_mask = torch.triu(torch.ones(1, attention_mask.size(-1), attention_mask.size(-1))) == 1
        sub_mask = sub_mask.to(attention_mask.device)
        mask = mask & sub_mask

    if expand_dims:
        mask = mask.repeat(num_attention_heads, 1, 1)

    mask = mask.bool().to(attention_mask.device)

    return mask


def generate_no_peek_mask(self, attention_mask) -> torch.Tensor:
    sz = attention_mask.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    mask = mask.to(self.device)
    return mask


def generate_padding_mask(self, seq, pad_idx):
    return (seq != pad_idx).to(self.device)
