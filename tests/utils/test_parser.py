def test_parser_args(parser_args):
    assert parser_args.dataset_variant == "arxiv"
    assert parser_args.batch_size == 4
    assert parser_args.stage == "fit"
    assert parser_args.dataset_limit == 32
    assert parser_args.model_name == "bert-base-uncased"
    assert parser_args.tokenizer_name == "bert-base-uncased"
    assert parser_args.frozen is True
    assert parser_args.learning_rate == 3e-4
    assert parser_args.num_workers == 0
    assert parser_args.bos_token == "<s>"
    assert parser_args.eos_token == "</s>"
