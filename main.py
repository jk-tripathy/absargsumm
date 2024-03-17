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
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import set_float32_matmul_precision

from models.GSum import GSum, GSumConfig
from utils import GenericDataModule, GenericModel, get_tokenizer, parser


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
        monitor="val loss",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=100,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    set_float32_matmul_precision("high")
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        print("Wandb API key not found")
        exit(1)
    wandb.login(key=api_key)

    args = parser()
    train(args)
