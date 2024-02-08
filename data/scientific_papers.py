import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from utils import GSumGuidance, get_tokenizer


class ScientificPapersDataset(Dataset):
    def __init__(self, args, split) -> None:
        self.tokenizer = get_tokenizer(args)
        self.gsum_guidance = GSumGuidance()
        self.dataset = load_dataset(
            "scientific_papers",
            name=args.dataset_variant,
            split=split,
            trust_remote_code=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # add bos and eos tokens
        article = f"{self.tokenizer.bos_token} {example['article']} {self.tokenizer.eos_token}"
        abstract = f"{self.tokenizer.bos_token} {example['abstract']}"
        guidance = self.gsum_guidance.get_guidance(article, abstract)

        tokenized_input = self.tokenizer(
            article,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized_guidance = self.tokenizer(
            guidance,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized_output = self.tokenizer(
            abstract,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {
            "input_ids": tokenized_input["input_ids"].flatten(),
            "attention_mask": tokenized_input["attention_mask"].flatten(),
            "guidance_input_ids": tokenized_guidance["input_ids"].flatten(),
            "guidance_attention_mask": tokenized_guidance["attention_mask"].flatten(),
            "decoder_input_ids": tokenized_output["input_ids"].flatten(),
            "decoder_attention_mask": tokenized_output["attention_mask"].flatten(),
        }


class ScientificPapersDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

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
            self.train_dataset = ScientificPapersDataset(self.args, split="train" + limit_length)
            self.val_dataset = ScientificPapersDataset(
                self.args, split="validation" + limit_length
            )
        elif stage == "validate":
            self.val_dataset = ScientificPapersDataset(
                self.args,
                split="validation" + limit_length,
            )
        elif stage == "test":
            self.test_dataset = ScientificPapersDataset(self.args, split="test" + limit_length)
        elif stage == "predict":
            self.predict_dataset = ScientificPapersDataset(self.args, split="test" + limit_length)

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
