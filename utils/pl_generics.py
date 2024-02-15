import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader

from data import GenericDataset


class GenericDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = GenericDataset(args)

    def setup(self, stage: str):
        """Setup the dataset for the given stage of the pipeline.

        Args:
            stage: Stage of the pipeline. Can be 'fit', 'validate', 'test', 'predict'
            dataset_limit: Limit the number of samples in the dataset. Defaults to None.
        """
        if self.args.dataset_limit is not None:
            limit_length = f"[:{self.args.dataset_limit}]"
        else:
            limit_length = ""

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(self.args, split="train" + limit_length)
            self.val_dataset = self.dataset(self.args, split="validation" + limit_length)
        elif stage == "validate":
            self.val_dataset = self.dataset(
                self.args,
                split="validation" + limit_length,
            )
        elif stage == "test":
            self.test_dataset = self.dataset(self.args, split="test" + limit_length)
        elif stage == "predict":
            self.predict_dataset = self.dataset(self.args, split="test" + limit_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )


class GenericModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        model_output = self(batch)
        self.log("train loss ", model_output.loss, prog_bar=True, logger=True)
        return model_output.loss

    def validation_step(self, batch, batch_idx):
        model_output = self(batch)
        self.log("val loss ", model_output.loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        model_output = self(batch)
        self.log("test loss ", model_output.loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
