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
from transformers import BertTokenizer

from utils.parser import parser


@pytest.fixture()
def parser_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--dataset_variant=arxiv",
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
    assert parser_args.frozen is True
    assert parser_args.learning_rate == 3e-4
    assert parser_args.num_workers == 0


@pytest.fixture
def batch_str():
    return [
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
    ]


@pytest.fixture()
def batch(batch_str):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    data = {
        "src": [],
        "src_attn_mask": [],
        "tgt": [],
        "tgt_attn_mask": [],
    }
    for str in batch_str:
        tokenized = tokenizer(
            str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        data["src"].append(tokenized["input_ids"].flatten())
        data["src_attn_mask"].append(tokenized["attention_mask"].flatten())
        data["tgt"].append(tokenized["input_ids"].flatten())
        data["tgt_attn_mask"].append(tokenized["attention_mask"].flatten())

    tensor_data = {
        "src": torch.stack(data["src"]),
        "src_attn_mask": torch.stack(data["src_attn_mask"]),
        "tgt": torch.stack(data["tgt"]),
        "tgt_attn_mask": torch.stack(data["tgt_attn_mask"]),
    }

    return tensor_data
