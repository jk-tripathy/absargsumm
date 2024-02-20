import pytest
import torch

from utils import GenericDataModule


@pytest.fixture
def datamodule(parser_args):
    dm = GenericDataModule(parser_args)
    return dm


@pytest.mark.parametrize("setup_stage", ["fit", "validate", "test"])
def test_dataset(datamodule, setup_stage):
    datamodule.setup(stage=setup_stage)
    if setup_stage == "fit":
        dataset = datamodule.train_dataset
    elif setup_stage == "validate":
        dataset = datamodule.val_dataset
    elif setup_stage == "test":
        dataset = datamodule.test_dataset

    example = dataset[0]

    assert example["input_ids"].shape == torch.Size([20])
    assert example["attention_mask"].shape == torch.Size([20])
    assert example["guidance_input_ids"].shape == torch.Size([20])
    assert example["guidance_attention_mask"].shape == torch.Size([20])
    assert example["decoder_input_ids"].shape == torch.Size([20])
    assert example["decoder_attention_mask"].shape == torch.Size([20])
