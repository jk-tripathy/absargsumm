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

import lightning.pytorch as pl
import wandb
from torch import set_float32_matmul_precision

from models.GSum import GSum, GSumConfig
from utils import GenericDataModule, GenericModel, parser

set_float32_matmul_precision("medium")


def train(args):
    dm = GenericDataModule(args)
    gsum_config = GSumConfig()
    gsum_model = GSum(gsum_config)

    model = GenericModel(gsum_model, args)
    logger = pl.loggers.WandbLogger(project=args.wandb_project, save_dir="logs")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        print("Wandb API key not found")
        exit(1)
    wandb.login(key=api_key)

    args = parser()
    train(args)
