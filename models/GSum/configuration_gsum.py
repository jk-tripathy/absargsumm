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
        freeze_encoder=False,
        dropout=0.2,
        encoder_learning_rate=0.002,
        decoder_learning_rate=0.2,
        warmup_steps=15000,
        bos_token="<s>",
        eos_token="</s>",
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
        self.encoder_learning_rate = encoder_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        self.warmup_steps = warmup_steps
        self.bos_token = bos_token
        self.eos_token = eos_token

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
            "encoder_learning_rate": self.encoder_learning_rate,
            "decoder_learning_rate": self.decoder_learning_rate,
            "warmup_steps": self.warmup_steps,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
