import json
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from configs.config import config


def load_json_data(split: str) -> list:
    """Load a dataset split from disk."""
    path = os.path.join(config.TRAINING_DATA_DIR, f"{split}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data


class SchemaDataset(Dataset):
    """PyTorch Dataset for schema column NER."""

    def __init__(self, split: str, tokenizer: BertTokenizerFast, max_length: int = None):
        self.data = load_json_data(split)
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_LENGTH
        self.label2id = config.LABEL2ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Tokenize with word IDs to align labels
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # special tokens ignored in loss
            elif word_idx != previous_word_idx:
                label = ner_tags[word_idx] if word_idx < len(ner_tags) else "O"
                aligned_labels.append(self.label2id.get(label, 0))
            else:
                # For subword tokens, use I- tag or ignore
                label = ner_tags[word_idx] if word_idx < len(ner_tags) else "O"
                aligned_labels.append(self.label2id.get(label, 0))
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def get_dataloaders(batch_size: int = None):
    """Return train, val, test dataloaders."""
    from torch.utils.data import DataLoader

    batch_size = batch_size or config.BATCH_SIZE

    tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)

    train_dataset = SchemaDataset("train", tokenizer)
    val_dataset = SchemaDataset("val", tokenizer)
    test_dataset = SchemaDataset("test", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    batch = next(iter(train_loader))
    print(f"\nSample batch keys: {list(batch.keys())}")
    print(f"Input IDs shape:   {batch['input_ids'].shape}")
    print(f"Labels shape:      {batch['labels'].shape}")