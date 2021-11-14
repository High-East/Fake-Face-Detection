import torch
from torch.utils.data import Dataset


class TmpDataset(Dataset):
    def __init__(self, TmpData):
        self.X = torch.rand(TmpData)
        self.y = torch.randint(high=2, size=TmpData[0:1])  # 0 or 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample
