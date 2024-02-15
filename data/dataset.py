import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from utils import GSumGuidance, get_tokenizer, parser


class GenericDataset(Dataset):
    def __init__(self, args, split) -> None:
        self.args = args
        self.tokenizer = get_tokenizer(args)
        if args.guidance != "none":
            self.guidance = GSumGuidance()
        else:
            self.guidance = None
        self.dataset = load_dataset(
            args.dataset,
            name=args.dataset_variant,
            split=split,
            trust_remote_code=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # add bos and eos tokens
        article = f"{self.tokenizer.bos_token} {example['article']} {self.tokenizer.eos_token}"
        abstract = f"{self.tokenizer.bos_token} {example['abstract']}"

        tokenized_input = self.tokenizer(
            article,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_input_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized_output = self.tokenizer(
            abstract,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_input_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ret_dict = {
            "input_ids": tokenized_input["input_ids"].flatten(),
            "attention_mask": tokenized_input["attention_mask"].flatten(),
            "decoder_input_ids": tokenized_output["input_ids"].flatten(),
            "decoder_attention_mask": tokenized_output["attention_mask"].flatten(),
        }

        if self.guidance is not None:
            guidance_signal = self.guidance.get_guidance(article, abstract)
            tokenized_signal = self.tokenizer(
                guidance_signal,
                padding="max_length",
                truncation=True,
                max_length=self.args.max_input_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            ret_dict["guidance_input_ids"] = tokenized_signal["input_ids"].flatten()
            ret_dict["guidance_attention_mask"] = tokenized_signal["attention_mask"].flatten()

        return ret_dict


if __name__ == "__main__":
    args = parser()
    guidance_file_path = f"data/{args.dataset}_{args.guidance}_guidance"
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

    guidance_dataset = load_from_disk(guidance_file_path)
    print(guidance_dataset.column_names)
