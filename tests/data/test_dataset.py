import pytest
import torch


@pytest.mark.parametrize("setup_stage", ["fit", "validate", "test"])
def test_dataset(dm, setup_stage):
    dm.setup(stage=setup_stage)
    if setup_stage == "fit":
        dataset = dm.train_dataset
    elif setup_stage == "validate":
        dataset = dm.val_dataset
    elif setup_stage == "test":
        dataset = dm.test_dataset

    example = dataset[0]

    assert example["input_ids"].shape == torch.Size([dm.tokenizer.model_max_length])
    assert example["attention_mask"].shape == torch.Size([dm.tokenizer.model_max_length])
    assert example["decoder_input_ids"].shape == torch.Size([dm.tokenizer.model_max_length])
    assert example["decoder_attention_mask"].shape == torch.Size([dm.tokenizer.model_max_length])
