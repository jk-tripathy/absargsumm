import pytest
import torch

from utils import GenericDataModule, parser


@pytest.fixture(params=["arxiv", "pubmed"])
def dataset_variant(request):
    return request.param


@pytest.fixture
def scientific_paper_args(monkeypatch, dataset_variant):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--dataset=scientific_papers",
            f"--dataset_variant={dataset_variant}",
            "--batch_size=4",
            "--stage=fit",
            "--dataset_limit=32",
            "--guidance=gsum",
        ],
    )
    args = parser()

    return args


@pytest.fixture
def datamodule(scientific_paper_args):
    dm = GenericDataModule(scientific_paper_args)
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

    example = dataset[0]

    assert example["input_ids"].shape == torch.Size([512])
    assert example["attention_mask"].shape == torch.Size([512])
    assert example["guidance_input_ids"].shape == torch.Size([512])
    assert example["guidance_attention_mask"].shape == torch.Size([512])
    assert example["decoder_input_ids"].shape == torch.Size([512])
    assert example["decoder_attention_mask"].shape == torch.Size([512])
