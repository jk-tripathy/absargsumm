import pytest

from models.SimpleTransformer.simple_trasformer_model import SimpleTransformer


@pytest.fixture()
def model(parser_args):
    model = SimpleTransformer(parser_args)
    return model


def test_model(model):
    assert model is not None


def test_forward(model, inputs, targets):
    output, loss = model(inputs, targets)
    assert output.shape == (2, 512)
    assert loss is not None
