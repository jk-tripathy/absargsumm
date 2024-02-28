import lightning.pytorch as pl
from evaluate import load
from torch import argmax, optim
from torch.utils.data import DataLoader

from data import GenericDataset
from utils import get_tokenizer


class GenericDataModule(pl.LightningDataModule):
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
            self.train_dataset = GenericDataset(self.args, split="train" + limit_length)
            self.val_dataset = GenericDataset(self.args, split="validation" + limit_length)
        elif stage == "validate":
            self.val_dataset = GenericDataset(self.args, split="validation" + limit_length)
        elif stage == "test":
            self.test_dataset = GenericDataset(self.args, split="test" + limit_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )


class GenericModel(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.tokenizer = get_tokenizer(self.args)

    def forward(self, batch):
        return self.model(**batch)

    def calculate_metrics(self, outputs, targets):
        rouge_metric = load("rouge")
        refs = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results = rouge_metric.compute(predictions=preds, references=refs)
        return results

    def training_step(self, batch, batch_idx):
        model_output = self(batch)
        self.log("train loss ", model_output.loss, prog_bar=True, logger=True)
        results = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        self.log("rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("rougeL", results["rougeL"], prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        model_output = self(batch)
        self.log("val loss ", model_output.loss, prog_bar=True, logger=True)
        results = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        self.log("rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("rougeL", results["rougeL"], prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        model_output = self(batch)
        self.log("test loss ", model_output.loss, prog_bar=True, logger=True)
        results = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        self.log("rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("rougeL", results["rougeL"], prog_bar=True, logger=True)

    def configure_optimizers(self):
        encoder_params = self.model.encoder.parameters()
        decoder_params = self.model.decoder.parameters()
        optimizer = optim.AdamW(
            [
                {"params": encoder_params, "lr": self.args.encoder_learning_rate},
                {"params": decoder_params, "lr": self.args.decoder_learning_rate},
            ]
        )

        return optimizer
