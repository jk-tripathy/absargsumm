import lightning.pytorch as pl
from evaluate import load
from torch import argmax, optim
from torch.utils.data import DataLoader
from transformers import get_inverse_sqrt_schedule

from data import GenericDataset
from utils import get_tokenizer


class GenericDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        dataset_variant,
        dataset_limit,
        longtext_column,
        shorttext_column,
        batch_size,
        tokenizer,
        guidance_type,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_variant = dataset_variant
        self.dataset_limit = dataset_limit
        self.longtext_column = longtext_column
        self.shorttext_column = shorttext_column
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.guidance_type = guidance_type

    def setup(self, stage: str):
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
            self.train_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="train" + limit_length,
                tokenizer=self.tokenizer,
                guidance_type=self.guidance_type,
            )
            self.val_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="validation" + limit_length,
                tokenizer=self.tokenizer,
                guidance_type=self.guidance_type,
            )

        elif stage == "validate":
            self.val_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="validation" + limit_length,
                tokenizer=self.tokenizer,
                guidance_type=self.guidance_type,
            )
        elif stage == "test":
            self.test_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="test" + limit_length,
                tokenizer=self.tokenizer,
                guidance_type=self.guidance_type,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


class GenericModel(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

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
        results = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        self.log("train loss", model_output.loss, prog_bar=True, logger=True)
        self.log("train rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("train rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("train rougeL", results["rougeL"], prog_bar=True, logger=True)
        return {
            "loss": model_output.loss,
            "train rouge1": results["rouge1"],
            "train rouge2": results["rouge2"],
            "train rougeL": results["rougeL"],
        }

    def validation_step(self, batch, batch_idx):
        model_output = self(batch)
        results = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        self.log("val loss", model_output.loss, prog_bar=True, logger=True)
        self.log("val rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("val rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("val rougeL", results["rougeL"], prog_bar=True, logger=True)
        return {
            "loss": model_output.loss,
            "val rouge1": results["rouge1"],
            "val rouge2": results["rouge2"],
            "val rougeL": results["rougeL"],
        }

    def test_step(self, batch, batch_idx):
        model_output = self(batch)
        results = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        self.log("test loss", model_output.loss, prog_bar=True, logger=True)
        self.log("test rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("test rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("test rougeL", results["rougeL"], prog_bar=True, logger=True)
        return {
            "loss": model_output.loss,
            "test rouge1": results["rouge1"],
            "test rouge2": results["rouge2"],
            "test rougeL": results["rougeL"],
        }

    def configure_optimizers(self):
        pretrained_encoder_params = self.model.encoder.pretrained_hf_encoder.parameters()
        source_transformer_layer = self.model.encoder.source_transformer_layer.parameters()
        guidance_transformer_layer = self.model.encoder.guidance_transformer_layer.parameters()
        decoder_params = self.model.decoder.parameters()

        optimizer = optim.Adam(
            [
                {
                    "params": pretrained_encoder_params,
                    "lr": self.model.config.encoder_learning_rate,
                },
                {
                    "params": source_transformer_layer,
                    "lr": self.model.config.decoder_learning_rate,
                },
                {
                    "params": guidance_transformer_layer,
                    "lr": self.model.config.decoder_learning_rate,
                },
                {
                    "params": decoder_params,
                    "lr": self.model.config.decoder_learning_rate,
                },
            ]
        )

        lr_scheduler = get_inverse_sqrt_schedule(optimizer, self.model.config.warmup_steps)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
