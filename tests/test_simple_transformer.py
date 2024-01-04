import pytest
from transformers import BertTokenizer

from models.SimpleTransformer.simple_trasformer_model import SimpleTransformer


@pytest.fixture()
def model(parser_args):
    model = SimpleTransformer(parser_args)
    return model


@pytest.fixture
def batch_str():
    return [
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
    ]


@pytest.fixture()
def inputs(batch_str):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    tokenized = tokenizer(
        batch_str,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    inputs = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "token_type_ids": tokenized["token_type_ids"],
    }
    return inputs


@pytest.fixture()
def targets(batch_str):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    tokenized = tokenizer(
        batch_str,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    target = tokenized["input_ids"]
    return target


def test_model(model):
    assert model is not None


def test_forward(model, inputs, targets):
    output, loss = model(inputs, targets)
    assert output.shape == (2, 512)
    assert loss is not None
