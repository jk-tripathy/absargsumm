import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="Abstractive Summarization of Scientific Papers using Argumentative Structure",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        help="AbsArgSumm experiment to run. Can be 'baseline', 'text_spans', 'annotated_spans', 'annotated_text'. Defaults to 'baseline'.",
    )
    parser.add_argument(
        "--guided",
        type=bool,
        default=False,
        help="Whether to use guided LED. Defaults to False.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Name of the Weights and Biases project. Defaults to None.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Type of accelerator to use. Defaults to 'gpu'.",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="Log and validate every n steps. Defaults to 100.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=-1,
        help="Maximum number of epochs. Defaults to -1.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of epochs. Defaults to -1.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=float,
        default=0,
        help="learning rate. defaults to 0.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scientific_papers",
        help="Name of the dataset to use. Defaults to 'scientific_papers'.",
    )
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="arxiv",
        help="Dataset variant of the Scientific Papers dataset. Choose between 'arxiv' and 'pubmed'",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=None,
        help="Limit the number of samples in the dataset. Useful for debugging. Defaults to None.",
    )
    parser.add_argument(
        "--shorttext_column",
        type=str,
        default=None,
        help="Column name of the short text in the dataset. Defaults to None.",
    )
    parser.add_argument(
        "--longtext_column",
        type=str,
        default=None,
        help="Column name of the long text in the dataset. Defaults to None.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size. Defaults to 16.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="fit",
        help="Stage of the pipeline. Can be 'fit', 'validate', 'test', 'predict'",
    )
    parser.add_argument(
        "--guidance_type",
        type=str,
        default="gsum",
        help="Guidance type to use. Defaults to 'gsum'.",
    )
    return parser.parse_args()
