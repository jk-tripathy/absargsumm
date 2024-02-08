import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data import ScientificPapersDataset


class GenericDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.DATASET_CLASSES_DICT = {
            "scientific_papers": ScientificPapersDataset,
        }
        self.dataset_class = self.DATASET_CLASSES_DICT[args.dataset]

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
            self.train_dataset = self.dataset_class(self.args, split="train" + limit_length)
            self.val_dataset = self.dataset_class(self.args, split="validation" + limit_length)
        elif stage == "validate":
            self.val_dataset = self.dataset_class(
                self.args,
                split="validation" + limit_length,
            )
        elif stage == "test":
            self.test_dataset = self.dataset_class(self.args, split="test" + limit_length)
        elif stage == "predict":
            self.predict_dataset = self.dataset_class(self.args, split="test" + limit_length)

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
