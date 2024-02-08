import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from utils import get_tokenizer, shift_tokens_right


def test_tokenizer(tokenizer):
    assert isinstance(tokenizer, PreTrainedTokenizerFast) or isinstance(
        tokenizer, PreTrainedTokenizer
    )
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"

    test_input_str = f"{tokenizer.bos_token} Hello {tokenizer.eos_token}"
    test_output_str = f"{tokenizer.bos_token} Hi"

    input_ids = tokenizer(
        test_input_str,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=8,
    ).input_ids.flatten()
    decoder_input_ids = tokenizer(
        test_output_str,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=8,
    ).input_ids.flatten()
    assert input_ids.shape == torch.Size([8])
    assert input_ids.tolist() == [30522, 7592, 30523, 0, 0, 0, 0, 0]
    assert decoder_input_ids.shape == torch.Size([8])
    assert decoder_input_ids.tolist() == [30522, 7632, 0, 0, 0, 0, 0, 0]


def test_tokenizer_for_generation(tokenizer):
    test_input_str = f"{tokenizer.bos_token} Hello {tokenizer.eos_token}"
    input_ids = tokenizer(
        test_input_str,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=8,
    ).input_ids
    assert input_ids.tolist() == [[30522, 7592, 30523, 0, 0, 0, 0, 0]]
    decoder_input_ids = shift_tokens_right(
        input_ids, tokenizer.pad_token_id, tokenizer.eos_token_id
    ).flatten()
    assert decoder_input_ids.tolist() == [30523, 30522, 7592, 0, 0, 0, 0, 0]
