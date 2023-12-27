import pytest
import torch

from data.scientific_papers import ScientificPapersDataModule


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

    sample_input, sample_target = dataset[0]
    assert type(sample_input["input_ids"]) is torch.Tensor
    assert type(sample_input["attention_mask"]) is torch.Tensor
    assert type(sample_input["token_type_ids"]) is torch.Tensor
    assert type(sample_target) is torch.Tensor

    assert sample_input["input_ids"].shape == torch.Size([512])
    assert sample_input["attention_mask"].shape == torch.Size([512])
    assert sample_input["token_type_ids"].shape == torch.Size([512])
    assert sample_target.shape == torch.Size([512])


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
    inputs, targets = batch_data

    assert type(inputs["input_ids"]) is torch.Tensor
    assert type(inputs["attention_mask"]) is torch.Tensor
    assert type(inputs["token_type_ids"]) is torch.Tensor
    assert type(targets) is torch.Tensor

    assert inputs["input_ids"].shape == torch.Size([4, 512])
    assert inputs["attention_mask"].shape == torch.Size([4, 512])
    assert inputs["token_type_ids"].shape == torch.Size([4, 512])
    assert targets.shape == torch.Size([4, 512])
