import pytest

from models.SimpleTransformer import SimpleTransformer, SimpleTransformerConfig


@pytest.fixture()
def model(parser_args):
    config = SimpleTransformerConfig()
    model = SimpleTransformer(config)
    return model


@pytest.fixture()
def registered_model():
    SimpleTransformerConfig.register_for_auto_class()
    SimpleTransformer.register_for_auto_class("AutoModelForSeq2SeqLM")
    config = SimpleTransformerConfig()
    model = SimpleTransformer(config)
    return model


def test_model(model):
    assert model is not None


def test_registered_model(registered_model, tokenizer):
    assert registered_model is not None


def test_forward(model, batch):
    output = model(**batch)
    assert output.logits.shape == (2, 512, 30522)
    assert output.loss is not None


def test_generate(model, tokenizer):
    tokenized_input = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
    output = model.generate(
        tokenized_input,
        max_length=20,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    assert output is not None
    output_decoded = tokenizer.decode(output[0])
    assert output_decoded is not None
