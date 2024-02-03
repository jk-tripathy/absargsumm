from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ScientificPapersDataset(Dataset):
    def __init__(self, dataset_variant: str, tokenizer_name: str, split: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )
        self.dataset = load_dataset(
            "scientific_papers",
            name=dataset_variant,
            split=split,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        processed_input = self.tokenizer(
            example["article"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        processed_output = self.tokenizer(
            example["abstract"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        return {
            "input_ids": processed_input["input_ids"].flatten(),
            "attention_mask": processed_input["attention_mask"].flatten(),
            "decoder_input_ids": processed_output["input_ids"].flatten(),
            "decoder_attention_mask": processed_output["attention_mask"].flatten(),
        }


class ScientificPapersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_variant: str,
        batch_size: int,
        num_workers: int,
        tokenizer_name: str,
        dataset_limit: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_variant = dataset_variant
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.dataset_limit = dataset_limit

    def setup(
        self,
        stage: str,
    ):
        """Setup the dataset for the given stage of the pipeline.

        Args:
            stage: Stage of the pipeline. Can be 'fit', 'validate', 'test', 'predict'
            dataset_limit: Limit the number of samples in the dataset. Defaults to None.
        """
        if self.dataset_limit is not None:
            limit_length = f"[:{self.dataset_limit}]"
        else:
            limit_length = ""

        if stage == "fit" or stage is None:
            self.train_dataset = ScientificPapersDataset(
                dataset_variant=self.dataset_variant,
                tokenizer_name=self.tokenizer_name,
                split="train" + limit_length,
            )
            self.val_dataset = ScientificPapersDataset(
                dataset_variant=self.dataset_variant,
                tokenizer_name=self.tokenizer_name,
                split="validation" + limit_length,
            )
        elif stage == "validate":
            self.val_dataset = ScientificPapersDataset(
                dataset_variant=self.dataset_variant,
                tokenizer_name=self.tokenizer_name,
                split="validation" + limit_length,
            )
        elif stage == "test":
            self.test_dataset = ScientificPapersDataset(
                dataset_variant=self.dataset_variant,
                tokenizer_name=self.tokenizer_name,
                split="test" + limit_length,
            )
        elif stage == "predict":
            self.predict_dataset = ScientificPapersDataset(
                dataset_variant=self.dataset_variant,
                tokenizer_name=self.tokenizer_name,
                split="test" + limit_length,
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
