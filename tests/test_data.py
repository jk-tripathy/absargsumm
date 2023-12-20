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


def test_train_dataloader(datamodule):
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    assert len(train_dataloader) == 2
    assert train_dataloader.batch_size == 4

    batch_data = next(iter(train_dataloader))
    assert type(batch_data["input_ids"]) is torch.Tensor
