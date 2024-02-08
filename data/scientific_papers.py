from datasets import load_dataset
from torch.utils.data import Dataset

from utils import GSumGuidance, get_tokenizer


class ScientificPapersDataset(Dataset):
    def __init__(self, args, split) -> None:
        self.tokenizer = get_tokenizer(args)
        if args.guidance != "none":
            self.guidance = GSumGuidance()
        else:
            self.guidance = None
        self.dataset = load_dataset(
            "scientific_papers",
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
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized_output = self.tokenizer(
            abstract,
            padding="max_length",
            truncation=True,
            max_length=512,
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
                max_length=512,
                return_tensors="pt",
                add_special_tokens=False,
            )
            ret_dict["guidance_input_ids"] = tokenized_signal["input_ids"].flatten()
            ret_dict["guidance_attention_mask"] = tokenized_signal["attention_mask"].flatten()

        return ret_dict
