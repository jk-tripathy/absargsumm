import pytest

from models.SimpleTransformer.simple_trasformer_model import SimpleTransformer


@pytest.fixture()
def model(parser_args):
    model = SimpleTransformer(parser_args)
    return model


def test_model(model):
    assert model is not None


def test_forward(model, batch):
    output, loss = model(**batch)
    assert output.shape == (2, 512, 30522)
    assert loss is not None
