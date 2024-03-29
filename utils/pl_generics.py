import lightning.pytorch as pl
import nltk
from evaluate import load
from torch import argmax, optim
from torch.utils.data import DataLoader
from transformers import get_inverse_sqrt_schedule
from wandb import Table

from data import GenericDataset


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
            train_limit_length = f"[:{self.dataset_limit}]"
            val_test_limit_length = f"[:{self.dataset_limit//8}]"
        else:
            train_limit_length = ""
            val_test_limit_length = ""

        if stage == "fit" or stage is None:
            self.train_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="train" + train_limit_length,
                tokenizer=self.tokenizer,
                guidance_type=self.guidance_type,
            )
            self.val_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="validation" + val_test_limit_length,
                tokenizer=self.tokenizer,
                guidance_type=self.guidance_type,
            )

        elif stage == "validate":
            self.val_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="validation" + val_test_limit_length,
                tokenizer=self.tokenizer,
                guidance_type=self.guidance_type,
            )
        elif stage == "test":
            self.test_dataset = GenericDataset(
                dataset=self.dataset,
                dataset_variant=self.dataset_variant,
                longtext_column=self.longtext_column,
                shorttext_column=self.shorttext_column,
                split="test" + val_test_limit_length,
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
        super(GenericModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_table = Table(columns=["step", "loss", "guidance", "gold text", "pred text"])
        self.val_table = Table(columns=["step", "loss", "guidance", "gold text", "pred text"])
        self.metric = load("rouge")

    def forward(self, batch):
        return self.model(**batch)

    def calculate_metrics(self, outputs, targets):
        refs = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        refs = ["\n".join(nltk.sent_tokenize(ref)) for ref in refs]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        results = self.metric.compute(predictions=preds, references=refs)
        return results, refs, preds

    def training_step(self, batch, batch_idx):
        model_output = self(batch)
        results, refs, preds = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        if batch_idx % self.trainer.val_check_interval == 0:
            guidance = self.tokenizer.batch_decode(
                batch["guidance_input_ids"], skip_special_tokens=True
            )
            self.train_table.add_data(
                self.trainer.global_step,
                model_output.loss.item(),
                guidance[0],
                refs[0],
                preds[0],
            )
            new_table = Table(columns=self.train_table.columns, data=self.train_table.data)
            self.logger.experiment.log({"samples/train": new_table}, commit=False)

        self.log("train/loss", model_output.loss.item(), prog_bar=True, logger=True)
        self.log("train/rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("train/rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("train/rougeL", results["rougeL"], prog_bar=True, logger=True)
        return {
            "loss": model_output.loss,
            "train/rouge1": results["rouge1"],
            "train/rouge2": results["rouge2"],
            "train/rougeL": results["rougeL"],
        }

    def validation_step(self, batch, batch_idx):
        model_output = self(batch)
        results, refs, preds = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        if batch_idx % self.trainer.val_check_interval == 0:
            guidance = self.tokenizer.batch_decode(
                batch["guidance_input_ids"], skip_special_tokens=True
            )
            self.val_table.add_data(
                self.trainer.global_step,
                model_output.loss.item(),
                guidance[0],
                refs[0],
                preds[0],
            )
            new_table = Table(columns=self.val_table.columns, data=self.val_table.data)
            self.logger.experiment.log({"samples/val": new_table}, commit=False)

        self.log("val/loss", model_output.loss.item(), prog_bar=True, logger=True)
        self.log("val/rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("val/rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("val/rougeL", results["rougeL"], prog_bar=True, logger=True)
        return {
            "loss": model_output.loss,
            "val/rouge1": results["rouge1"],
            "val/rouge2": results["rouge2"],
            "val/rougeL": results["rougeL"],
        }

    def test_step(self, batch, batch_idx):
        model_output = self(batch)
        results, refs, preds = self.calculate_metrics(
            argmax(model_output.logits, dim=-1), batch["decoder_input_ids"]
        )
        self.log("test/loss", model_output.loss.item(), prog_bar=True, logger=True)
        self.log("test/rouge1", results["rouge1"], prog_bar=True, logger=True)
        self.log("test/rouge2", results["rouge2"], prog_bar=True, logger=True)
        self.log("test/rougeL", results["rougeL"], prog_bar=True, logger=True)
        return {
            "loss": model_output.loss,
            "test/rouge1": results["rouge1"],
            "test/rouge2": results["rouge2"],
            "test/rougeL": results["rougeL"],
        }

    def configure_optimizers(self):
        pretrained_encoder_params = self.model.encoder.pretrained_hf_encoder.parameters()
        source_transformer_layer = self.model.encoder.source_transformer_layer.parameters()
        guidance_transformer_layer = self.model.encoder.guidance_transformer_layer.parameters()
        decoder_params = self.model.decoder.parameters()

        optimizer = optim.AdamW(
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
