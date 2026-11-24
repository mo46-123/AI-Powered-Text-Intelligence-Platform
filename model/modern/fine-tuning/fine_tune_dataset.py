import pandas as pd 
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, file, tokenizer, max_length=256):
        self.source = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.source.iloc[idx, 3],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self.source.iloc[idx, 2]
        return item