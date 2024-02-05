import pytest
from models.pretrained_hf_model import PretrainedHFModel
from transformers import BertModel


@pytest.fixture(scope="module", params=[True, False])
def freeze_model(request):
    return request.param


@pytest.fixture(scope="module")
def base_model(freeze_model):
    return PretrainedHFModel("bert-base-uncased", frozen=freeze_model)


def test_load(base_model):
    assert base_model is not None
    assert isinstance(base_model.model, BertModel)


def test_output_shape(base_model, batch):
    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    outputs = base_model(inputs)
    assert outputs.shape == (2, 512, 768)


def test_frozen(base_model, freeze_model):
    assert base_model.frozen == freeze_model
    for param in base_model.model.parameters():
        assert param.requires_grad != freeze_model
