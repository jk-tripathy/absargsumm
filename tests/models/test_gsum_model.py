import pytest

from models.GSum import GSum, GSumConfig, GSumDecoder, GSumEncoder
from utils import create_masks


@pytest.fixture()
def encoder():
    config = GSumConfig()
    encoder = GSumEncoder(config)
    return encoder


@pytest.fixture()
def decoder():
    config = GSumConfig()
    decoder = GSumDecoder(config)
    return decoder


@pytest.fixture()
def model():
    config = GSumConfig()
    model = GSum(config)
    return model


def test_encoder(encoder):
    assert encoder is not None


def test_model(model):
    assert model is not None


def test_source_encoder(encoder, batch_with_guidance):
    output = encoder(**batch_with_guidance)
    assert output is not None
    assert output.source_last_hidden_state.shape == (4, 20, 768)
    assert output.guidance_last_hidden_state.shape == (4, 20, 768)


def test_decoder(encoder, decoder, batch_with_guidance):
    encoder_output = encoder(**batch_with_guidance)
    output = decoder(
        source_logits=encoder_output.source_last_hidden_state,
        guidance_logits=encoder_output.guidance_last_hidden_state,
        target_input_ids=batch_with_guidance["decoder_input_ids"],
    )

    assert output.shape == (4, 20, 768)


def test_forward_SANITYCHECK(model, batch_with_guidance):
    source_attentions = create_masks(
        batch_with_guidance["attention_mask"],
        expand_dims=True,
        num_attention_heads=12,
        for_causal=False,
    )
    target_attentions = create_masks(
        batch_with_guidance["decoder_attention_mask"],
        expand_dims=True,
        num_attention_heads=12,
        for_causal=True,
    )
    guidance_attentions = source_attentions.clone()
    encoder_output = model.encoder(**batch_with_guidance)
    assert encoder_output is not None
    decoder_output = model.decoder(
        source_logits=encoder_output.source_last_hidden_state,
        source_attentions=source_attentions,
        guidance_logits=encoder_output.guidance_last_hidden_state,
        guidance_attentions=guidance_attentions,
        target_input_ids=batch_with_guidance["decoder_input_ids"],
        target_attentions=target_attentions,
    )
    assert decoder_output is not None
    logits = model.linear(decoder_output)
    assert logits.shape == (4, 20, 30524)
    loss = model.loss(logits.permute(0, 2, 1), batch_with_guidance["decoder_input_ids"])
    assert loss is not None


# TODO:
# these two are probably failing due to HF inheritance
# figure out why
def test_forward(model, batch_with_guidance):
    output = model(**batch_with_guidance)
    assert output.loss is not None
    assert output.logits.shape == (4, 20, 30524)


def test_generate(model, batch_with_guidance, tokenizer):
    tokenized_input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    tokenized_guidance = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = model.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        guidance_input_ids=tokenized_guidance["input_ids"],
        guidance_attention_mask=tokenized_guidance["attention_mask"],
        max_length=50,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    assert output is not None
    output_decoded = tokenizer.decode(output[0])
    assert output_decoded is not None
