import pandas as pd 
from torch.utils.data import Dataset, DataLoader

class ReviewDataset(Dataset):
    def __init__(self, source_csv):
        self.data = pd.read_csv(source_csv)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data.iloc[key, 3], self.data.iloc[key, 2]