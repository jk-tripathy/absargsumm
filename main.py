import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

import lightning.pytorch as pl
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
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    args = parser()
    train(args)
