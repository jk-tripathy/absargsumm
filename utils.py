import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="Abstractive Summarization of Scientific Papers using Argumentative Structure",
    )
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="arxiv",
        help="Dataset variant of the Scientific Papers dataset. Choose between 'arxiv' and 'pubmed'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size. Defaults to 16.",
    )
    # num_workers is broken and setting it to a positive value breaks the dataloader.
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers. Defaults to 0.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="dev",
        help="Stage of the pipeline. Can be 'fit', 'validate', 'test', 'predict'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the model and tokenizer to use. Defaults to 'bert-base-uncased'.",
    )
    parser.add_argument(
        "--frozen",
        type=bool,
        default=True,
        help="Whether to freeze the base model. Defaults to True.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate. Defaults to 3e-4.",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=None,
        help="Limit the number of samples in the dataset. Useful for debugging. Defaults to None.",
    )
    return parser.parse_args()
