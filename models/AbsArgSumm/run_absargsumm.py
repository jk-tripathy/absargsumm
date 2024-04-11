from evaluate import load
from transformers import AutoModelForSeq2SeqLM

from data import SciArg
from models.AbsArgSumm import GuidedLEDForConditionalGeneration
from utils import get_tokenizer


class AbsArgSumm:
    def __init__(self, experiment="baseline", guided=False):
        self.experiment = experiment
        self.model_name = "allenai/led-large-16384-arxiv"
        self.batch_size = 2
        self.max_input_length = 8192
        self.max_output_length = 512
        self.rouge = load("rouge")
        if guided:
            if experiment == "baseline":
                raise ValueError("Guided LED not supported for baseline experiment")

            self.model = GuidedLEDForConditionalGeneration.from_pretrained(
                gradient_checkpointing=True,
                use_cache=False,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                gradient_checkpointing=True,
                use_cache=False,
            )
        if experiment == "baseline" or experiment == "text_spans":
            self.tokenizer = get_tokenizer(self.model_name)
        elif experiment == "annotated_text":
            special_tokens = ["<ADU>", "</ADU>"]
            self.tokenizer = get_tokenizer(self.model_name, special_tokens=special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.data = SciArg(
            tokenizer=self.tokenizer,
            experiment=self.experiment,
            batch_size=self.batch_size,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
        )

        self._post_init()

    def _post_init(self):
        self.model.config.num_beams = 2
        self.model.config.max_length = 512
        self.model.config.min_length = 100
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = self.rouge.compute(predictions=pred_str, references=label_str)

        return {
            "rouge1": round(rouge_output["rouge1"], 4),
            "rouge2": round(rouge_output["rouge2"], 4),
            "rougeL": round(rouge_output["rougeL"], 4),
        }
