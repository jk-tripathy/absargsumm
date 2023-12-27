import pytest
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer

from models.base_model import BaseModel


@pytest.fixture(scope="module", params=[True, False])
def freeze_model(request):
    return request.param


@pytest.fixture(scope="module")
def base_model(freeze_model):
    return BaseModel("bert-base-uncased", frozen=freeze_model)


@pytest.fixture
def input_str():
    return "Hello, my dog is cute"


@pytest.fixture
def input_batch_str():
    return [
        "Hello, my dog is cute",
        "Hello, my cat is also cute",
    ]


def test_load(base_model):
    assert base_model is not None
    assert isinstance(base_model.model, BertModel)


def test_output_shape(base_model, input_str, input_batch_str):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    input_ids = tokenizer(
        input_str,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    outputs = base_model(input_ids)
    assert outputs.shape == (1, 512, 768)

    input_ids = tokenizer(
        input_batch_str,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    outputs = base_model(input_ids)
    assert outputs.shape == (2, 512, 768)


def test_frozen(base_model, freeze_model):
    assert base_model.frozen == freeze_model
    for param in base_model.model.parameters():
        assert param.requires_grad != freeze_model
