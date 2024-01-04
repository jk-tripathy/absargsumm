import pytest

from models.GSum.gsum_model import GSum


@pytest.fixture()
def model(parser_args):
    model = GSum(parser_args)
    return model


def test_model(model):
    assert model is not None


def test_source_encoder(model, inputs):
    output = model._source_encoder(inputs)
    assert output.shape == (2, 512, 768)


def test_guidance_encoder(model, inputs):
    output = model._guidance_encoder(inputs)
    assert output.shape == (2, 512, 768)


def test_decoder(model, inputs, targets):
    source_output = model._source_encoder(inputs)
    guidance_output = model._guidance_encoder(inputs)
    target_embeded = model.target_embed(targets)
    output = model.output_decoder(
        source=source_output,
        source_mask=None,
        guidance=guidance_output,
        guidance_mask=None,
        target=target_embeded,
        target_mask=None,
    )
    assert output.shape == (2, 512, 768)


def test_forward(model, inputs, targets):
    output, loss = model(inputs, targets)
    assert output.shape == (2, 512)
    assert loss is not None
