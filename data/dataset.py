import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import GSumGuidance, parser


class GenericDataset(Dataset):
    def __init__(
        self,
        dataset,
        dataset_variant,
        longtext_column,
        shorttext_column,
        split,
        tokenizer,
        guidance_type,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.longtext_column = longtext_column
        self.shorttext_column = shorttext_column
        self.dataset = load_dataset(
            dataset,
            name=dataset_variant,
            split=split,
            trust_remote_code=True,
        )

        dirty_samples = []
        if guidance_type != "none":
            guidance_datafiles = {
                "train": f"processed_guidance/{dataset}_{guidance_type}_guidance/train/train.arrow",
                "validation": f"processed_guidance/{dataset}_{guidance_type}_guidance/validation/validation.arrow",
                "test": f"processed_guidance/{dataset}_{guidance_type}_guidance/test/test.arrow",
            }
            self.guidance_dataset = load_dataset(
                "arrow", data_files=guidance_datafiles, split=split
            )
            for idx, sample in tqdm(
                enumerate(zip(self.dataset, self.guidance_dataset)),
                total=len(self.dataset),
                desc=f"Checking for empty samples in {split}",
            ):
                data, guidance = sample
                if (
                    data[self.longtext_column] == ""
                    or data[self.shorttext_column] == ""
                    or guidance["guidance"] == ""
                ):
                    dirty_samples.append(idx)
            self.guidance_dataset = self.guidance_dataset.select(
                (i for i in range(len(self.guidance_dataset)) if i not in set(dirty_samples))
            )
        else:
            self.guidance_dataset = None
            for idx, data in tqdm(
                enumerate(self.dataset),
                desc=f"Checking for empty samples in {split}",
            ):
                if data[self.longtext_column] == "" or data[self.shorttext_column] == "":
                    dirty_samples.append(idx)

        self.dataset = self.dataset.select(
            (i for i in range(len(self.dataset)) if i not in set(dirty_samples))
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        article = example[self.longtext_column]
        abstract = example[self.shorttext_column]

        tokenized_input = self.tokenizer(
            article,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokenized_output = self.tokenizer(
            abstract,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        ret_dict = {
            "input_ids": tokenized_input["input_ids"].flatten(),
            "attention_mask": tokenized_input["attention_mask"].flatten(),
            "decoder_input_ids": tokenized_output["input_ids"].flatten(),
            "decoder_attention_mask": tokenized_output["attention_mask"].flatten(),
        }

        if self.guidance_dataset is not None:
            guidance_signal = self.guidance_dataset[idx]["guidance"]
            tokenized_signal = self.tokenizer(
                guidance_signal,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            ret_dict["guidance_input_ids"] = tokenized_signal["input_ids"].flatten()
            ret_dict["guidance_attention_mask"] = tokenized_signal["attention_mask"].flatten()

        return ret_dict


if __name__ == "__main__":
    args = parser()
    guidance_file_path = f"processed_guidance/{args.dataset}_{args.guidance}_guidance"
    dataset = load_dataset(args.dataset, name=args.dataset_variant)
    col_names = dataset["train"].column_names
    guidance = GSumGuidance()
    updated_dataset = dataset.map(
        lambda example: {
            "guidance": guidance.get_guidance(
                example[args.longtext_column],
                example[args.shorttext_column],
            )
        },
        batched=False,
        remove_columns=col_names,
    )
    updated_dataset.save_to_disk(guidance_file_path)
