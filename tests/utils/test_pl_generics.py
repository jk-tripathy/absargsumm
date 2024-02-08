import pytest
import torch

from utils import GenericDataModule


@pytest.fixture
def datamodule(parser_args):
    dm = GenericDataModule(parser_args)
    return dm


@pytest.mark.parametrize("setup_stage", ["fit", "validate", "test", "predict"])
def test_dataloader(datamodule, setup_stage):
    datamodule.setup(stage=setup_stage)
    if setup_stage == "fit":
        dataloader = datamodule.train_dataloader()
    elif setup_stage == "validate":
        dataloader = datamodule.val_dataloader()
    elif setup_stage == "test":
        dataloader = datamodule.test_dataloader()
    elif setup_stage == "predict":
        dataloader = datamodule.predict_dataloader()

    batch_data = next(iter(dataloader))

    assert type(batch_data["input_ids"]) is torch.Tensor
    assert type(batch_data["attention_mask"]) is torch.Tensor
    assert type(batch_data["guidance_input_ids"]) is torch.Tensor
    assert type(batch_data["guidance_attention_mask"]) is torch.Tensor
    assert type(batch_data["decoder_input_ids"]) is torch.Tensor
    assert type(batch_data["decoder_attention_mask"]) is torch.Tensor

    assert batch_data["input_ids"].shape == torch.Size([4, 512])
    assert batch_data["attention_mask"].shape == torch.Size([4, 512])
    assert batch_data["guidance_input_ids"].shape == torch.Size([4, 512])
    assert batch_data["guidance_attention_mask"].shape == torch.Size([4, 512])
    assert batch_data["decoder_input_ids"].shape == torch.Size([4, 512])
    assert batch_data["decoder_attention_mask"].shape == torch.Size([4, 512])
