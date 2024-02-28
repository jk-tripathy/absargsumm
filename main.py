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
from utils import GenericDataModule, GenericModel, parser


def train(args):
    dm = GenericDataModule(args)
    gsum_config = GSumConfig()
    gsum_model = GSum(gsum_config)

    model = GenericModel(gsum_model, args)
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
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    set_float32_matmul_precision("medium")
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        print("Wandb API key not found")
        exit(1)
    wandb.login(key=api_key)

    args = parser()
    train(args)
