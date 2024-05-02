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
        "--shared_encoder",
        type=bool,
        default=False,
        help="Whether to share the encoder betweent the source and the guidance. Defaults to False.",
    )
    return parser.parse_args()
