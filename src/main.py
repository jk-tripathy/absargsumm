from data import ScientificPapersDataModule
from utils import parser

if __name__ == "__main__":
    args = parser()

    dm = ScientificPapersDataModule(
        dataset_variant=args.dataset_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name=args.tokenizer_name,
    )
    dm.setup(stage=args.stage)
    print(dm.dev_dataset)
