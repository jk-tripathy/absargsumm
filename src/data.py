import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class ScientificPapersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_variant: str,
        batch_size: int,
        num_workers: int,
        tokenizer_name: str,
    ):
        super().__init__()
        self.dataset_variant = dataset_variant
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(self, example):
        processed_data = self.tokenizer(example["article"], return_tensors="pt")
        processed_data["labels"] = self.tokenizer(
            example["abstract"],
            return_tensors="pt",
        )["input_ids"]
        return processed_data

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = load_dataset(
                "scientific_papers", name=self.dataset_variant, split="train"
            )
            self.train_dataset = self.train_dataset.map(
                self.preprocess_function, batched=True, batch_size=self.batch_size
            )
            self.val_dataset = load_dataset(
                "scientific_papers", name=self.dataset_variant, split="validation"
            )
            self.val_dataset = self.val_dataset.map(
                self.preprocess_function, batched=True, batch_size=self.batch_size
            )
        elif stage == "test":
            self.test_dataset = load_dataset(
                "scientific_papers", name=self.dataset_variant, split="test"
            )
            self.test_dataset = self.test_dataset.map(
                self.preprocess_function, batched=True, batch_size=self.batch_size
            )
        elif stage == "predict":
            self.predict_dataset = load_dataset(
                "scientific_papers", name=self.dataset_variant, split="test"
            )
            self.predict_dataset = self.predict_dataset.map(
                self.preprocess_function, batched=True, batch_size=self.batch_size
            )
        elif stage == "dev":
            self.dev_dataset = load_dataset(
                "scientific_papers", name=self.dataset_variant, split="train[:100]"
            )
            self.dev_dataset = self.dev_dataset.map(
                self.preprocess_function, batched=True, batch_size=self.batch_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def dev_dataloader(self):
        return DataLoader(
            self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
