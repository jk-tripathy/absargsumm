from xml.etree.ElementTree import ElementTree, fromstring

from datasets import load_dataset


class SciArg:
    def __init__(self, tokenizer, batch_size=2, max_input_length=8192, max_output_length=512):
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer

        self.dataset = load_dataset("DFKI-SLT/sciarg", split="train").train_test_split(
            test_size=0.1
        )
        self.train_val_dataset, self.test_dataset = self.dataset["train"], self.dataset["test"]
        self.train_val_dataset = self.train_val_dataset.train_test_split(test_size=0.02)
        self.train_dataset, self.val_dataset = (
            self.train_val_dataset["train"],
            self.train_val_dataset["test"],
        )

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
        self.val_dataset = self.val_dataset.map(
            self._process_data_to_model_inputs,
            batched=True,
            batch_size=self.batch_size,
        )
        self.val_dataset.set_format(
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
            text = doc
            tree = ElementTree(fromstring(text))
            root = tree.getroot()
            text = ""
            abs = ""
            for child in root:
                if child.tag == "Abstract":
                    abs = child.text
                elif child.tag.startswith("H"):
                    for paragraph in child:
                        text += paragraph.text
            full_text.append(text)
            abstract.append(abs)
        return full_text, abstract

    def _process_data_to_model_inputs(self, batch):
        full_text, abstract = self._parse_xml(batch["text"])

        # tokenize the inputs and labels
        inputs = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
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
