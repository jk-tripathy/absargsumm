import pytest
import torch

from src.data import ScientificPapersDataModule


@pytest.fixture(scope="module", params=["arxiv", "pubmed"])
def dataset_variant(request):
    return request.param


@pytest.fixture(scope="module")
def datamodule(dataset_variant):
    dm = ScientificPapersDataModule(
        dataset_variant=dataset_variant,
        batch_size=4,
        num_workers=0,
        tokenizer_name="bert-base-uncased",
        dataset_limit=8,
    )
    return dm


@pytest.mark.parametrize("setup_stage", ["fit", "validate", "test", "predict"])
def test_dataset(datamodule, setup_stage):
    datamodule.setup(stage=setup_stage)
    if setup_stage == "fit":
        dataset = datamodule.train_dataset
    elif setup_stage == "validate":
        dataset = datamodule.val_dataset
    elif setup_stage == "test":
        dataset = datamodule.test_dataset
    elif setup_stage == "predict":
        dataset = datamodule.predict_dataset

    sample = dataset[0]
    assert type(sample["input_ids"]) is torch.Tensor
    assert type(sample["attention_mask"]) is torch.Tensor
    assert type(sample["token_type_ids"]) is torch.Tensor
    assert type(sample["target"]) is torch.Tensor

    assert sample["input_ids"].shape == torch.Size([512])
    assert sample["attention_mask"].shape == torch.Size([512])
    assert sample["token_type_ids"].shape == torch.Size([512])
    assert sample["target"].shape == torch.Size([512])


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
    assert type(batch_data["token_type_ids"]) is torch.Tensor
    assert type(batch_data["target"]) is torch.Tensor

    assert batch_data["input_ids"].shape == torch.Size([4, 512])
    assert batch_data["attention_mask"].shape == torch.Size([4, 512])
    assert batch_data["token_type_ids"].shape == torch.Size([4, 512])
    assert batch_data["target"].shape == torch.Size([4, 512])
