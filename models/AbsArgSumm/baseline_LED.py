"""Checking the results of pretrained LED model on SciArg dataset as baseline.

This experiment uses full text as input and the abstract as output.
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data import SciArg


class BaselineLED:
    def __init__(self):
        self.model_name = "allenai/led-large-16384-arxiv"
        self.batch_size = 2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            gradient_checkpointing=True,
            use_cache=False,
        )
        self.rouge = load("rouge")
        self.data = SciArg(self.tokenizer, batch_size=self.batch_size)

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

        rouge_output = self.rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }
