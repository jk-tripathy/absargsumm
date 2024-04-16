import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

import os
from datetime import datetime

import lightning.pytorch as pl
import nltk
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import set_float32_matmul_precision
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from data import GenericDataModule
from models import GenericModel
from models.AbsArgSumm import AbsArgSumm
from models.GSum import GSum, GSumConfig
from utils import get_tokenizer, parser


def train(args):
    gsum_config = GSumConfig()
    gsum_model = GSum(gsum_config)
    tokenizer = get_tokenizer(
        model_name=gsum_config.pretrained_encoder_name_or_path,
        bos_token=gsum_config.bos_token,
        eos_token=gsum_config.eos_token,
    )
    dm = GenericDataModule(
        dataset=args.dataset,
        dataset_variant=args.dataset_variant,
        dataset_limit=args.dataset_limit,
        longtext_column=args.longtext_column,
        shorttext_column=args.shorttext_column,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        guidance_type=args.guidance_type,
    )

    model = GenericModel(model=gsum_model, tokenizer=tokenizer)
    formatted_timedate = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logger = pl.loggers.WandbLogger(project=args.wandb_project, save_dir="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"saved_models/{formatted_timedate}",
        save_top_k=1,
        monitor="val/loss",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        log_every_n_steps=args.log_step,
        val_check_interval=args.log_step,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dm)


def test(args):
    gsum_config = GSumConfig()
    gsum_model = GSum(gsum_config)
    tokenizer = get_tokenizer(
        model_name=gsum_config.pretrained_encoder_name_or_path,
        bos_token=gsum_config.bos_token,
        eos_token=gsum_config.eos_token,
    )
    dm = GenericDataModule(
        dataset=args.dataset,
        dataset_variant=args.dataset_variant,
        dataset_limit=args.dataset_limit,
        longtext_column=args.longtext_column,
        shorttext_column=args.shorttext_column,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        guidance_type=args.guidance_type,
    )

    model = GenericModel.load_from_checkpoint(
        "saved_models/2024-03-19_13-30/epoch=4-step=560.ckpt",
        model=gsum_model,
        tokenizer=tokenizer,
    )
    logger = pl.loggers.WandbLogger(project=args.wandb_project, save_dir="logs")
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        log_every_n_steps=args.log_step,
        val_check_interval=args.log_step,
    )
    trainer.test(model, dm)


def AbsArgSummExperiments(experiment: str, guided: bool, shared_encoder: bool = False):
    """
    Run experiments for AbsArgSumm
    Args:
        experiment (str): experiment to run. Can be one of ["baseline", "text_spans", "annotated_text"]
        guided (bool): whether to use guided LED
    """
    if guided and not shared_encoder:
        os.environ["WANDB_PROJECT"] = f"GuidedAbsArgSumm_{experiment}"
    elif guided and shared_encoder:
        os.environ["WANDB_PROJECT"] = f"SharedGuidedAbsArgSumm_{experiment}"
    else:
        os.environ["WANDB_PROJECT"] = f"AbsArgSumm_{experiment}"

    run = AbsArgSumm(experiment=experiment, guided=guided, shared_encoder=shared_encoder)
    # enable fp16 apex training
    formatted_timedate = datetime.now().strftime("%Y-%m-%d_%H-%M")
    training_args = Seq2SeqTrainingArguments(
        seed=42,
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=run.batch_size,
        per_device_eval_batch_size=run.batch_size,
        fp16=True,
        output_dir=f"logs/AbsArgSumm/{experiment}/{formatted_timedate}",
        logging_steps=5,
        eval_steps=10,
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        num_train_epochs=300,
        report_to="wandb",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Seq2SeqTrainer(
        model=run.model,
        tokenizer=run.tokenizer,
        args=training_args,
        compute_metrics=run.compute_metrics,
        train_dataset=run.data.train_dataset,
        eval_dataset=run.data.test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    pl.seed_everything(42)
    set_float32_matmul_precision("high")
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        print("Wandb API key not found")
        exit(1)
    wandb.login(key=api_key)

    # nltk.download("punkt")

    args = parser()
    AbsArgSummExperiments(
        experiment=args.experiment,
        guided=args.guided,
        shared_encoder=args.shared_encoder,
    )
