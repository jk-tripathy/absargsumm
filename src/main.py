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

from data import ScientificPapersDataModule
from models.gsum import GSum
from utils import parser


def train(args):
    dm = ScientificPapersDataModule(
        dataset_variant=args.dataset_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name=args.model_name,
        dataset_limit=args.dataset_limit,
    )
    model = GSum(args)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    args = parser()
    train(args)
