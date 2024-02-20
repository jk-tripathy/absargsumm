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

from utils import get_tokenizer, parser


@pytest.fixture()
def parser_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--dataset=cnn_dailymail",
            "--dataset_variant=3.0.0",
            "--shorttext_column=highlights",
            "--longtext_column=article",
            "--batch_size=4",
            "--stage=fit",
            "--dataset_limit=32",
            "--max_input_length=20",
            "--guidance=gsum",
        ],
    )
    args = parser()

    return args


@pytest.fixture
def tokenizer(parser_args):
    return get_tokenizer(parser_args)


@pytest.fixture
def batch_str():
    return [
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
    ]


@pytest.fixture()
def batch(batch_str, tokenizer, parser_args):
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
            max_length=parser_args.max_input_length,
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


@pytest.fixture()
def batch_with_guidance(batch):
    batch["guidance_input_ids"] = batch["input_ids"].clone()
    batch["guidance_attention_mask"] = batch["attention_mask"].clone()
    return batch
