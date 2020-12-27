import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class StructGraphDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


dataset = StructGraphDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)