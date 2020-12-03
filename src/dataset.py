import dgl.data
from torch.utils.data import DataLoader
import torch

class StructDataset():

    def __init__(self, dataset_name="MUTAG"):
        self.dataset = dgl.data.GINDataset(name=dataset_name)
        self.dataset = list(self.dataset)
        self.train_dataset = self.dataset[:int(len(self.dataset) * 0.8)]
        self.test_dataset = self.dataset[int(len(self.dataset))*0.8:]

    def collate(self, sample):
        graphs, labels = map(list, zip(*sample))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

    def get_train_dev_data_loader(self, batch_size=1024):
        train_dataset = self.train_dataset[:int(len(self.dataset) * 0.8)]
        dev = self.train_dataset[int(len(self.dataset) * 0.8):]
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=self.collate,
            drop_last=False,
            shuffle=True)
        dev_dataloader = DataLoader(
            dev,
            batch_size=batch_size,
            collate_fn=self.collate,
            drop_last=False,
            shuffle=True)
        return train_dataloader, dev_dataloader

    def get_test_data_loader(self, batch_size=120):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            collate_fn=self.collate,
            drop_last=False,
            shuffle=False)
        return test_dataloader