import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

import pytest

from utils import parser


@pytest.fixture()
def parser_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--dataset_variant=arxiv",
            "--batch_size=4",
            "--stage=fit",
            "--dataset_limit=32",
        ],
    )
    args = parser()

    return args


def test_parser_args(parser_args):
    assert parser_args.dataset_variant == "arxiv"
    assert parser_args.batch_size == 4
    assert parser_args.stage == "fit"
    assert parser_args.dataset_limit == 32
    assert parser_args.model_name == "bert-base-uncased"
    assert parser_args.frozen is True
    assert parser_args.learning_rate == 3e-4
    assert parser_args.num_workers == 0
