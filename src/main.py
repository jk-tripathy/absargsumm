from data import ScientificPapersDataModule
from utils import parser

if __name__ == "__main__":
    args = parser()

    dm = ScientificPapersDataModule(
        dataset_variant=args.dataset_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name=args.model_name,
    )
    dm.setup(stage=args.stage, dataset_limit=args.dataset_limit)
    print(dm.train_dataset)
    print(dm.val_dataset)
