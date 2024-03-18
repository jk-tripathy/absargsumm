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

from models.GSum import GSumConfig
from utils import GenericDataModule, get_tokenizer, parser


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
            "--dataset_limit=32",
            "--guidance_type=none",
        ],
    )
    args = parser()

    return args


@pytest.fixture
def tokenizer():
    config = GSumConfig()
    return get_tokenizer(
        model_name=config.pretrained_encoder_name_or_path,
        bos_token=config.bos_token,
        eos_token=config.eos_token,
    )


@pytest.fixture
def dm(parser_args, tokenizer):
    dm = GenericDataModule(
        dataset=parser_args.dataset,
        dataset_variant=parser_args.dataset_variant,
        dataset_limit=parser_args.dataset_limit,
        longtext_column=parser_args.longtext_column,
        shorttext_column=parser_args.shorttext_column,
        batch_size=parser_args.batch_size,
        guidance_type=parser_args.guidance_type,
        tokenizer=tokenizer,
    )
    return dm


@pytest.fixture
def batch_str():
    return [
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
    ]


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
            max_length=tokenizer.model_max_length,
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
