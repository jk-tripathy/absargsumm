import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="Abstractive Summarization of Scientific Papers using Argumentative Structure",
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
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs. Defaults to 1.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of epochs. Defaults to 1.",
    )
    parser.add_argument(
        "--encoder_warmup_steps",
        type=int,
        default=20000,
        help="Number of warmup steps for the encoder. Defaults to 10000.",
    )
    parser.add_argument(
        "--decoder_warmup_steps",
        type=int,
        default=10000,
        help="Number of warmup steps for the decoder. Defaults to 10000.",
    )
    parser.add_argument(
        "--encoder_learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate. Defaults to 3e-4.",
    )
    parser.add_argument(
        "--decoder_learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate. Defaults to 3e-4.",
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
        default="fit",
        help="Stage of the pipeline. Can be 'fit', 'validate', 'test', 'predict'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the model to use. Defaults to 'bert-base-uncased'.",
    )
    parser.add_argument(
        "--frozen",
        type=bool,
        default=True,
        help="Whether to freeze the base model. Defaults to True.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the tokenizer to use. Defaults to 'bert-base-uncased'.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=512,
        help="Maximum input length. Defaults to 512.",
    )
    parser.add_argument(
        "--bos_token",
        type=str,
        default="<s>",
        help="Beginning of sentence token. Defaults to '<s>'.",
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default="</s>",
        help="End of sentence token. Defaults to '</s>'.",
    )
    parser.add_argument(
        "--batch_first",
        type=bool,
        default=True,
        help="Whether to freeze the base model. Defaults to True.",
    )
    parser.add_argument(
        "--guidance",
        type=str,
        default="gsum",
        help="Guidance type to use. Defaults to 'gsum'.",
    )
    return parser.parse_args()
