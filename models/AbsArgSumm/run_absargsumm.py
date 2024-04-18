import json
from datetime import datetime

from evaluate import load
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data import SciArg
from models.AbsArgSumm import GuidedLEDForConditionalGeneration
from utils import get_tokenizer


class AbsArgSumm:
    def __init__(
        self,
        experiment="baseline",
        guided=False,
        shared_encoder=False,
        project_name="AbsArgSumm",
    ):
        self.experiment = experiment
        self.guided = guided
        self.shared_encoder = shared_encoder
        self.project_name = project_name
        self.model_name = "allenai/led-large-16384-arxiv"
        self.batch_size = 2
        self.max_input_length = 8192
        self.max_output_length = 512
        self.rouge = load("rouge")

        if experiment == "baseline" or experiment == "text_spans":
            self.tokenizer = get_tokenizer(self.model_name)
        elif experiment == "annotated_text" or experiment == "annotated_spans":
            special_tokens = ["<ADU>", "</ADU>"]
            self.tokenizer = get_tokenizer(self.model_name, special_tokens=special_tokens)

        self.data = SciArg(
            tokenizer=self.tokenizer,
            experiment=self.experiment,
            guided=self.guided,
            batch_size=self.batch_size,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
        )

    def _post_init(self):
        self.model.config.num_beams = 2
        self.model.config.max_length = 512
        self.model.config.min_length = 100
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3

    def model_init(self):
        if self.guided:
            if self.experiment == "baseline":
                raise ValueError("Guided LED not supported for baseline experiment")

            config = AutoConfig.from_pretrained(self.model_name)
            config.gradient_checkpointing = True
            config.use_cache = False
            config.shared_encoder = self.shared_encoder
            self.model = GuidedLEDForConditionalGeneration(config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                gradient_checkpointing=True,
                use_cache=False,
            )

        if self.experiment == "annotated_text" or self.experiment == "annotated_spans":
            self.model.resize_token_embeddings(len(self.tokenizer))

        self._post_init()

        return self.model

    def train(self):
        # enable fp16 apex training
        self.formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            fp16=True,
            output_dir=f"logs/{self.project_name}/{self.formatted_datetime}",
            logging_steps=5,
            eval_steps=10,
            save_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            gradient_accumulation_steps=4,
            num_train_epochs=300,
            report_to="wandb",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.trainer = Seq2SeqTrainer(
            model_init=self.model_init,
            tokenizer=self.tokenizer,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.data.train_dataset,
            eval_dataset=self.data.test_dataset,
        )

        self.trainer.train()

    def evaluate(self):
        eval_results = self.trainer.evaluate()
        return eval_results

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

    def multirun(self, num_runs=3):
        multirun_results = {"average": {"rouge1": 0, "rouge2": 0, "rougeL": 0, "loss": 0}}
        for i in range(num_runs):
            # enable fp16 apex training
            self.formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
            self.training_args = Seq2SeqTrainingArguments(
                seed=i,
                predict_with_generate=True,
                evaluation_strategy="steps",
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                fp16=True,
                output_dir=f"logs/{self.project_name}/{self.formatted_datetime}",
                logging_steps=5,
                eval_steps=10,
                save_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="rougeL",
                gradient_accumulation_steps=4,
                num_train_epochs=300,
                report_to="wandb",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.trainer = Seq2SeqTrainer(
                model_init=self.model_init,
                tokenizer=self.tokenizer,
                args=self.training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=self.data.train_dataset,
                eval_dataset=self.data.test_dataset,
            )
            self.trainer.train()
            eval_results = self.trainer.evaluate()

            eval_results["output_dir"] = self.training_args.output_dir
            multirun_results[f"run_{i+1}"] = eval_results
            for metric in ["rouge1", "rouge2", "rougeL", "loss"]:
                multirun_results["average"][metric] += eval_results[f"eval_{metric}"]

        for metric in ["rouge1", "rouge2", "rougeL", "loss"]:
            multirun_results["average"][metric] /= num_runs

        multirun_file = f"logs/{self.project_name}/multirun_results_{self.formatted_datetime}.json"
        with open(multirun_file, "w") as f:
            json.dump(multirun_results, f)
        print(f"Averaged Results: {multirun_results['average']}")
        print(f"Multirun results saved to {multirun_file}")
