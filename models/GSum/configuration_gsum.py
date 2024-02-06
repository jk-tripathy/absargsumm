from transformers import PretrainedConfig


class GSumConfig(PretrainedConfig):
    model_type = "gsum"

    def __init__(
        self,
        vocab_size=30524,
        d_model=768,
        decoder_ff_dim=2048,
        nhead=12,
        num_decoder_layers=12,
        batch_first=True,
        pretrained_encoder_name_or_path="bert-base-uncased",
        freeze_encoder=True,
        dropout=0.1,
        **kwargs,
    ):
        super(GSumConfig, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.decoder_ff_dim = decoder_ff_dim
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.batch_first = batch_first
        self.pretrained_encoder_name_or_path = pretrained_encoder_name_or_path
        self.freeze_encoder = freeze_encoder
        self.dropout = dropout

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "decoder_ff_dim": self.decoder_ff_dim,
            "nhead": self.nhead,
            "batch_first": self.batch_first,
            "pretrained_encoder_name_or_path": self.pretrained_encoder_name_or_path,
            "freeze_encoder": self.freeze_encoder,
            "dropout": self.dropout,
        }
