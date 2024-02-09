import numpy as np
import torch.utils.data


class DataframeToDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.data = list(df["data"])
        self.labels = list(df["label"])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]

        label = self.labels[idx]
        # label = torch.tensor(label, dtype=torch.long)
        return idx,data, label


class ListtoDataset(torch.utils.data.Dataset):
    def __init__(self, lista):
        self.data = lista
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        # label = torch.tensor(label, dtype=torch.long)
        return data