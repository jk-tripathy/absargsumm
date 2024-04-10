import lightning.pytorch as pl
import nltk
from evaluate import load
from torch import argmax, optim
from transformers import get_inverse_sqrt_schedule

from wandb import Table


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
