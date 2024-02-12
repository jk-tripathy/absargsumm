import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

import pytorch_lightning as pl

from models.GSum import GSum, GSumConfig
from utils import GenericDataModule, GenericModel, parser


def train(args):
    dm = GenericDataModule(args)
    gsum_config = GSumConfig()
    gsum_model = GSum(gsum_config)
    model = GenericModel(gsum_model)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=10,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    args = parser()
    train(args)
