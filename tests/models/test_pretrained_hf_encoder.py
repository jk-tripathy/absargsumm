import pytest
from transformers import BertModel

from models import PretrainedHFEncoder


@pytest.fixture(scope="module", params=[True, False])
def freeze_model(request):
    return request.param


@pytest.fixture(scope="module")
def base_model(freeze_model):
    return PretrainedHFEncoder("bert-base-uncased", frozen=freeze_model)


def test_load(base_model):
    assert base_model is not None
    assert isinstance(base_model.model, BertModel)


def test_output_shape(base_model, batch):
    outputs = base_model(**batch)
    assert outputs.last_hidden_state.shape == (2, 512, 768)


def test_frozen(base_model, freeze_model):
    assert base_model.frozen == freeze_model
    for param in base_model.model.parameters():
        assert param.requires_grad != freeze_model
