import pytest
import torch

from models.GSum import GSum, GSumConfig
from utils import GenericModel


@pytest.fixture
def model(tokenizer):
    config = GSumConfig()
    model = GSum(config)
    model = GenericModel(model, tokenizer)
    return model


@pytest.mark.parametrize("setup_stage", ["fit", "validate", "test"])
def test_dataloader(dm, setup_stage, batch_with_guidance):
    dm.setup(stage=setup_stage)
    if setup_stage == "fit":
        dataloader = dm.train_dataloader()
    elif setup_stage == "validate":
        dataloader = dm.val_dataloader()
    elif setup_stage == "test":
        dataloader = dm.test_dataloader()

    batch = next(iter(dataloader))
    batch["guidance_input_ids"] = batch["input_ids"]
    batch["guidance_attention_mask"] = batch["attention_mask"]

    assert type(batch["input_ids"]) is torch.Tensor
    assert type(batch["attention_mask"]) is torch.Tensor
    assert type(batch["guidance_input_ids"]) is torch.Tensor
    assert type(batch["guidance_attention_mask"]) is torch.Tensor
    assert type(batch["decoder_input_ids"]) is torch.Tensor
    assert type(batch["decoder_attention_mask"]) is torch.Tensor

    assert batch["input_ids"].shape == torch.Size([4, dm.tokenizer.model_max_length])
    assert batch["attention_mask"].shape == torch.Size([4, dm.tokenizer.model_max_length])
    assert batch["guidance_input_ids"].shape == torch.Size([4, dm.tokenizer.model_max_length])
    assert batch["guidance_attention_mask"].shape == torch.Size([4, dm.tokenizer.model_max_length])
    assert batch["decoder_input_ids"].shape == torch.Size([4, dm.tokenizer.model_max_length])
    assert batch["decoder_attention_mask"].shape == torch.Size([4, dm.tokenizer.model_max_length])

    assert batch.keys() == batch_with_guidance.keys()


def test_calculate_metrics(model, batch):
    results, _, _ = model.calculate_metrics(batch["decoder_input_ids"], batch["decoder_input_ids"])
    assert results["rouge1"] == 1.0
    assert results["rouge2"] == 1.0
    assert results["rougeL"] == 1.0
