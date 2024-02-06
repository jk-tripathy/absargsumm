from transformers import PretrainedConfig


class SimpleTransformerConfig(PretrainedConfig):
    model_type = "simple_transformer"

    def __init__(
        self,
        vocab_size=30524,
        hidden_size=768,
        num_attention_heads=12,
        batch_first=True,
        **kwargs,
    ):
        super(SimpleTransformerConfig, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.batch_first = batch_first

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "batch_first": self.batch_first,
        }
