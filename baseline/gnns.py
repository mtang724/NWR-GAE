from dgl.nn.pytorch import GraphConv, GINConv, GATConv
import torch
import torch.nn as nn
from dgl.data import CitationGraphDataset
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from data import build_graph, utils
from src.layers import MLP

import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = GATConv(hidden_dim * num_heads, hidden_dim, 1)

    def forward(self, g, input):
        h = self.layer1(g, input)
        # h = torch.relu(h)
        # h = self.layer2(g, h)
        return h

class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h


# self.linear1 = MLP(2, in_feats, hidden_size, hidden_size)
#         self.graphconv1 = GINConv(apply_func=self.linear1, aggregator_type='sum')
class GIN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GIN, self).__init__()
        self.linear1 = MLP(2, in_feats, hidden_size, hidden_size)
        self.conv1 = GINConv(apply_func=self.linear1, aggregator_type='sum')

        self.linear2 = MLP(2, hidden_size, hidden_size, hidden_size)
        self.conv2 = GINConv(apply_func=self.linear2, aggregator_type='sum')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

dataset = CitationGraphDataset("cora")
g = dataset[0]
node_features = g.ndata['feat']
node_labels = g.ndata['label']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

g, labels = utils.read_real_datasets("wisconsin")
node_features = g.ndata['attr']
node_labels = labels
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

acc = 0
for i in range(10):
    # node_embeddings = train(g, node_features, lr=1e-4, epoch=25, device=device, encoder="GCN")
    # torch.save(node_embeddings.cpu().detach(), 'embeddings.pt')
    # node_embeddings = torch.load("embeddings.pt")
    gc = GCN(n_features, n_features, n_labels)
    node_embeddings = gc(g, node_features)
    input_dims = node_embeddings.shape
    class_number = int(max(node_labels)) + 1
    FNN = MLP(num_layers=5, input_dim=input_dims[1], hidden_dim=input_dims[1]//2, output_dim=class_number)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(FNN.parameters())

    dataset = NodeClassificationDataset(node_embeddings, node_labels)
    split = utils.DataSplit(dataset, shuffle=True)
    train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
    # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    best = -float('inf')
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # data = data.to(device)
            inputs, labels = data
            y_pred = FNN(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                correct = 0
                total = 0
                for data in val_loader:
                    inputs, labels = data
                    outputs = FNN(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels)
            if correct / total > best:
                best = correct / total
                torch.save(FNN.state_dict(), 'best_mlp.pkl')
            print(str(epoch), correct / total)
    with torch.no_grad():
        FNN.load_state_dict(torch.load('best_mlp.pkl'))
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            outputs = FNN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    print((correct / total).item())
    acc += (correct / total).item()
print(acc/10)