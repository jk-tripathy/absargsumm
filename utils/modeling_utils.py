from typing import List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(
    model_name: str,
    special_tokens: Optional[List[str]] = None,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if special_tokens is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    return tokenizer
