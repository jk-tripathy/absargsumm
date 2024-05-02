import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

import json
from statistics import mean, median, stdev

from pie_datasets import load_dataset
from pie_modules.documents import (
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from utils import get_tokenizer


class SciArg:
    def __init__(
        self,
        tokenizer,
        experiment,
        guided,
        batch_size,
        max_input_length,
        max_output_length,
        doEDA=False,
    ):
        self.tokenizer = tokenizer
        self.experiment = experiment
        self.guided = guided
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.raw_dataset = load_dataset("pie/sciarg", split="train")
        self.dataset = self.raw_dataset.to_document_type(
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        ).train_test_split(test_size=0.2)

        self.train_dataset, self.test_dataset = self.dataset["train"], self.dataset["test"]

        if doEDA:
            self._eda()
        else:
            self._post_init()

    def _eda(self):
        self.eda = {"train": {}, "test": {}}

        def helper(len_list):
            return {
                "min": min(len_list),
                "max": max(len_list),
                "mean": round(mean(len_list), 2),
                "std": round(stdev(len_list), 2),
            }

        full_text_lens = []
        abstract_lens = []
        text_spans_lens = []
        annotated_spans_lens = []
        annotated_full_texts_lens = []
        for x in self.train_dataset:
            full_text, abstract = self._parse_xml([x])
            text_spans, annotated_spans, annotated_full_texts = self._parse_annotations(
                [x], full_text
            )
            full_text_lens.append(len(full_text[0].split()))
            abstract_lens.append(len(abstract[0].split()))
            text_spans_lens.append(len(text_spans[0].split()))
            annotated_spans_lens.append(len(annotated_spans[0].split()))
            annotated_full_texts_lens.append(len(annotated_full_texts[0].split()))

        self.eda["train"]["full_text"] = helper(full_text_lens)
        self.eda["train"]["abstract"] = helper(abstract_lens)
        self.eda["train"]["text_spans"] = helper(text_spans_lens)
        self.eda["train"]["annotated_spans"] = helper(annotated_spans_lens)
        self.eda["train"]["annotated_full_texts"] = helper(annotated_full_texts_lens)

        full_text_lens = []
        abstract_lens = []
        text_spans_lens = []
        annotated_spans_lens = []
        annotated_full_texts_lens = []
        for x in self.test_dataset:
            full_text, abstract = self._parse_xml([x])
            text_spans, annotated_spans, annotated_full_texts = self._parse_annotations(
                [x], full_text
            )
            full_text_lens.append(len(full_text[0].split()))
            abstract_lens.append(len(abstract[0].split()))
            text_spans_lens.append(len(text_spans[0].split()))
            annotated_spans_lens.append(len(annotated_spans[0].split()))
            annotated_full_texts_lens.append(len(annotated_full_texts[0].split()))

        self.eda["test"]["full_text"] = helper(full_text_lens)
        self.eda["test"]["abstract"] = helper(abstract_lens)
        self.eda["test"]["text_spans"] = helper(text_spans_lens)
        self.eda["test"]["annotated_spans"] = helper(annotated_spans_lens)
        self.eda["test"]["annotated_full_texts"] = helper(annotated_full_texts_lens)

    def _post_init(self):
        if self.guided:
            self.train_dataset = self.train_dataset.map(
                self._process_guided_data_to_model_inputs,
                batched=True,
                batch_size=self.batch_size,
            )
            self.test_dataset = self.test_dataset.map(
                self._process_guided_data_to_model_inputs,
                batched=True,
                batch_size=self.batch_size,
            )
            self.train_dataset.set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "guidance_input_ids",
                    "guidance_attention_mask",
                    "global_attention_mask",
                    "labels",
                ],
            )
            self.test_dataset.set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "guidance_input_ids",
                    "guidance_attention_mask",
                    "global_attention_mask",
                    "labels",
                ],
            )
        else:
            self.train_dataset = self.train_dataset.map(
                self._process_data_to_model_inputs,
                batched=True,
                batch_size=self.batch_size,
            )
            self.test_dataset = self.test_dataset.map(
                self._process_data_to_model_inputs,
                batched=True,
                batch_size=self.batch_size,
            )
            self.train_dataset.set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "global_attention_mask",
                    "labels",
                ],
            )
            self.test_dataset.set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "global_attention_mask",
                    "labels",
                ],
            )

    def _parse_xml(self, batch):
        full_text = []
        abstract = []
        for doc in batch:
            text = ""
            abs = ""
            for partition in doc.labeled_partitions:
                if partition.label == "Abstract":
                    abs = str(partition)
                else:
                    text += str(partition)
            full_text.append(text)
            abstract.append(abs)
        return full_text, abstract

    def _parse_annotations(self, batch, full_texts, adu_start="<ADU>", adu_end="</ADU>"):
        text_spans = []
        annotated_text_spans = []
        annotated_full_texts = []
        for full_text, doc in zip(full_texts, batch):
            text_span = ""
            annotated_text_span = ""
            annotated_full_text = full_text
            seen_spans = set()
            for annotation in doc.metadata["span_texts"]:
                if len(annotation.split()) > 2 and annotation not in seen_spans:
                    seen_spans.add(annotation)
                    text_span += annotation + " "
                    annotated_text_span += adu_start + " " + annotation + " " + adu_end + " "
                    annotated_full_text = annotated_full_text.replace(
                        annotation, adu_start + " " + annotation + adu_end + " "
                    )
            text_spans.append(text_span)
            annotated_text_spans.append(annotated_text_span)
            annotated_full_texts.append(annotated_full_text)
        return text_spans, annotated_text_spans, annotated_full_texts

    def _process_data_to_model_inputs(self, batch):
        full_text, abstract = self._parse_xml(batch)
        text_spans, annotated_spans, annotated_full_texts = self._parse_annotations(
            batch, full_text
        )

        if self.experiment == "baseline":
            inputs = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
            )
        elif self.experiment == "text_spans":
            inputs = self.tokenizer(
                text_spans,
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
            )
        elif self.experiment == "annotated_spans":
            inputs = self.tokenizer(
                annotated_spans,
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
                add_special_tokens=True,
            )
        elif self.experiment == "annotated_text":
            inputs = self.tokenizer(
                annotated_full_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
                add_special_tokens=True,
            )

        outputs = self.tokenizer(
            abstract,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_length,
        )

        batch = {}
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    def _process_guided_data_to_model_inputs(self, batch):
        full_text, abstract = self._parse_xml(batch)
        text_spans, annotated_spans, annotated_full_texts = self._parse_annotations(
            batch, full_text
        )

        inputs = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )

        if self.experiment == "text_spans":
            guidance_inputs = self.tokenizer(
                text_spans,
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
            )
        elif self.experiment == "annotated_spans":
            guidance_inputs = self.tokenizer(
                annotated_spans,
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
                add_special_tokens=True,
            )
        elif self.experiment == "annotated_text":
            guidance_inputs = self.tokenizer(
                annotated_full_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
                add_special_tokens=True,
            )

        outputs = self.tokenizer(
            abstract,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_length,
        )

        batch = {}
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["guidance_input_ids"] = guidance_inputs.input_ids
        batch["guidance_attention_mask"] = guidance_inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch


if __name__ == "__main__":
    tokenizer = get_tokenizer(model_name="allenai/led-large-16384-arxiv")
    dataset = SciArg(
        tokenizer=tokenizer,
        experiment="baseline",
        guided=False,
        batch_size=2,
        max_input_length=8195,
        max_output_length=512,
        doEDA=True,
    )
    print(json.dumps(dataset.eda, indent=4))
