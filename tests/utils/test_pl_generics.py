import pytest
import pytorch_lightning as pl
import torch

from models.SimpleTransformer import SimpleTransformer, SimpleTransformerConfig
from utils import GenericDataModule, GenericModel


@pytest.fixture
def datamodule(parser_args):
    dm = GenericDataModule(parser_args)
    return dm


@pytest.fixture
def model():
    config = SimpleTransformerConfig()
    simple_model = SimpleTransformer(config)
    model = GenericModel(simple_model)
    return model


@pytest.mark.parametrize("setup_stage", ["fit", "validate", "test"])
def test_dataloader(datamodule, setup_stage, batch_with_guidance):
    datamodule.setup(stage=setup_stage)
    if setup_stage == "fit":
        dataloader = datamodule.train_dataloader()
    elif setup_stage == "validate":
        dataloader = datamodule.val_dataloader()
    elif setup_stage == "test":
        dataloader = datamodule.test_dataloader()

    batch_data = next(iter(dataloader))

    assert type(batch_data["input_ids"]) is torch.Tensor
    assert type(batch_data["attention_mask"]) is torch.Tensor
    assert type(batch_data["guidance_input_ids"]) is torch.Tensor
    assert type(batch_data["guidance_attention_mask"]) is torch.Tensor
    assert type(batch_data["decoder_input_ids"]) is torch.Tensor
    assert type(batch_data["decoder_attention_mask"]) is torch.Tensor

    assert batch_data["input_ids"].shape == torch.Size([4, 20])
    assert batch_data["attention_mask"].shape == torch.Size([4, 20])
    assert batch_data["guidance_input_ids"].shape == torch.Size([4, 20])
    assert batch_data["guidance_attention_mask"].shape == torch.Size([4, 20])
    assert batch_data["decoder_input_ids"].shape == torch.Size([4, 20])
    assert batch_data["decoder_attention_mask"].shape == torch.Size([4, 20])

    assert batch_data.keys() == batch_with_guidance.keys()


def test_model_forward(datamodule, model):
    datamodule.setup(stage="fit")
    batch_data = next(iter(datamodule.train_dataloader()))
    output = model.forward(batch_data)
    assert output.loss is not None


# def test_fit(datamodule, model):
#     trainer = pl.Trainer(
#         accelerator="cpu",
#         max_epochs=1,
#     )
#     trainer.fit(model, datamodule)
#     assert True
