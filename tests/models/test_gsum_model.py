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
    assert output["input_embeds"].shape == (2, 512, 768)
    assert output["source_logits"].shape == (2, 512, 768)
    assert output["guidance_input_embeds"].shape == (2, 512, 768)
    assert output["guidance_logits"].shape == (2, 512, 768)


def test_decoder(encoder, decoder, batch_with_guidance):
    encoder_output = encoder(**batch_with_guidance)
    output = decoder(
        source_logits=encoder_output["source_logits"],
        source_attentions=None,
        guidance_logits=encoder_output["guidance_logits"],
        guidance_attentions=None,
        target_input_ids=batch_with_guidance["decoder_input_ids"],
        target_attentions=None,
    )

    assert output.shape == (2, 512, 768)


def test_forward_SANITYCHECK(model, batch_with_guidance):
    source_attentions, target_attentions = create_masks(
        12, batch_with_guidance["attention_mask"], batch_with_guidance["decoder_attention_mask"]
    )
    guidance_attentions = source_attentions.clone()
    encoder_output = model.encoder(**batch_with_guidance)
    assert encoder_output is not None
    decoder_output = model.decoder(
        source_logits=encoder_output["source_logits"],
        source_attentions=source_attentions,
        guidance_logits=encoder_output["guidance_logits"],
        guidance_attentions=guidance_attentions,
        target_input_ids=batch_with_guidance["decoder_input_ids"],
        target_attentions=target_attentions,
    )
    assert decoder_output is not None
    logits = model.linear(decoder_output)
    assert logits.shape == (2, 512, 30524)
    loss = model.loss(logits.permute(0, 2, 1), batch_with_guidance["decoder_input_ids"])
    assert loss is not None


# TODO:
# these two are probably failing due to HF inheritance
# figure out why
# def test_forward(model, batch_with_guidance):
#     output = model(**batch_with_guidance)
#     assert output.loss is not None
#     assert output.logits.shape == (2, 512, 30524)
#
#
# def test_generate(model, tokenizer):
#     tokenized_input = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
#     output = model.generate(
#         tokenized_input,
#         max_length=20,
#         num_beams=2,
#         no_repeat_ngram_size=2,
#         early_stopping=True,
#     )
#     assert output is not None
#     output_decoded = tokenizer.decode(output[0])
#     assert output_decoded is not None
