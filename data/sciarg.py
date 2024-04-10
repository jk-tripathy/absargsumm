# from datasets import load_dataset
from pie_datasets import load_dataset
from pie_modules.documents import (
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)


class SciArg:
    def __init__(
        self,
        tokenizer,
        experiment,
        batch_size,
        max_input_length,
        max_output_length,
    ):
        self.experiment = experiment
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer

        self.raw_dataset = load_dataset("pie/sciarg", split="train")
        self.dataset = self.raw_dataset.to_document_type(
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        ).train_test_split(test_size=0.2)

        self.train_dataset, self.test_dataset = self.dataset["train"], self.dataset["test"]

        self._post_init()

    def _post_init(self):
        self.train_dataset = self.train_dataset.map(
            self._process_data_to_model_inputs,
            batched=True,
            batch_size=self.batch_size,
        )
        self.train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
        )
        self.test_dataset = self.test_dataset.map(
            self._process_data_to_model_inputs,
            batched=True,
            batch_size=self.batch_size,
        )
        self.test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
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
        annotated_full_texts = []
        for full_text, doc in zip(full_texts, batch):
            text_span = ""
            annotated_full_text = full_text
            seen_spans = set()
            for annotation in doc.metadata["span_texts"]:
                if len(annotation.split()) > 2 and annotation not in seen_spans:
                    seen_spans.add(annotation)
                    text_span += annotation + " "
                    annotated_full_text = annotated_full_text.replace(
                        annotation, adu_start + annotation + adu_end
                    )
            text_spans.append(text_span)
            annotated_full_texts.append(annotated_full_text)
        return text_spans, annotated_full_texts

    def _process_data_to_model_inputs(self, batch):
        full_text, abstract = self._parse_xml(batch)
        text_spans, annotated_full_texts = self._parse_annotations(batch, full_text)

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
