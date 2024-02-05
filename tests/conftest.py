import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

import pytest
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from utils import get_tokenizer, parser, shift_tokens_right


@pytest.fixture()
def parser_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--batch_size=4",
            "--stage=fit",
            "--dataset_limit=32",
        ],
    )
    args = parser()

    return args


def test_parser_args(parser_args):
    assert parser_args.dataset_variant == "arxiv"
    assert parser_args.batch_size == 4
    assert parser_args.stage == "fit"
    assert parser_args.dataset_limit == 32
    assert parser_args.model_name == "bert-base-uncased"
    assert parser_args.tokenizer_name == "bert-base-uncased"
    assert parser_args.frozen is True
    assert parser_args.learning_rate == 3e-4
    assert parser_args.num_workers == 0
    assert parser_args.bos_token == "<s>"
    assert parser_args.eos_token == "</s>"


@pytest.fixture
def batch_str():
    return [
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
    ]


@pytest.fixture
def tokenizer(parser_args):
    return get_tokenizer(parser_args)


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


@pytest.fixture()
def batch(batch_str, tokenizer):
    data = {
        "input_ids": [],
        "attention_mask": [],
        "decoder_input_ids": [],
        "decoder_attention_mask": [],
    }
    for str in batch_str:
        tokenized = tokenizer(
            str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        data["input_ids"].append(tokenized["input_ids"].flatten())
        data["attention_mask"].append(tokenized["attention_mask"].flatten())
        data["decoder_input_ids"].append(tokenized["input_ids"].flatten())
        data["decoder_attention_mask"].append(tokenized["attention_mask"].flatten())

    tensor_data = {
        "input_ids": torch.stack(data["input_ids"]),
        "attention_mask": torch.stack(data["attention_mask"]),
        "decoder_input_ids": torch.stack(data["decoder_input_ids"]),
        "decoder_attention_mask": torch.stack(data["decoder_attention_mask"]),
    }

    return tensor_data
