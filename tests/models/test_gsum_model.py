import pytest

from models.GSum import GSum, GSumConfig
from utils import create_masks


@pytest.fixture()
def model():
    config = GSumConfig()
    model = GSum(config)
    return model


def test_model(model):
    assert model is not None


# def test_forward(model, batch_with_guidance):
#     output = model(**batch_with_guidance)
#     assert output.loss is not None
#     assert output.logits.shape == (4, 512, 30524)
#
#
# def test_generate(model, batch_with_guidance, tokenizer):
#     tokenized_input = tokenizer(
#         "Hello, my dog is cute",
#         return_tensors="pt",
#         max_length=10,
#         truncation=True,
#         padding="max_length",
#     )
#     tokenized_guidance = tokenizer(
#         "Hello, my dog is cute",
#         return_tensors="pt",
#         max_length=10,
#         truncation=True,
#         padding="max_length",
#     )
#     output = model.generate(
#         input_ids=tokenized_input["input_ids"],
#         attention_mask=tokenized_input["attention_mask"],
#         guidance_input_ids=tokenized_guidance["input_ids"],
#         guidance_attention_mask=tokenized_guidance["attention_mask"],
#         max_length=50,
#         num_beams=2,
#         no_repeat_ngram_size=2,
#         early_stopping=True,
#     )
#     assert output is not None
#     output_decoded = tokenizer.decode(output[0])
#     assert output_decoded is not None
