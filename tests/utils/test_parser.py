def test_parser_args(parser_args):
    assert parser_args.dataset == "cnn_dailymail"
    assert parser_args.dataset_variant == "3.0.0"
    assert parser_args.shorttext_column == "highlights"
    assert parser_args.longtext_column == "article"
    assert parser_args.guidance_type == "none"
    assert parser_args.batch_size == 4
    assert parser_args.dataset_limit == 32
