from src.layers import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import GINConv
import dgl

from src.utils import FocalLoss
import scipy
import scipy.optimize
import torch.multiprocessing as mp


# FNN with gumbal_softmax
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(x)
        x = F.gumbel_softmax(x)
        return x


def chamfer_loss(predictions, targets, mask):
    if mask == 0:
        return 0
    predictions = predictions[:mask, :]
    targets = targets[:mask, :]
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    # predictions and targets shape :: (n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = (predictions - targets).pow(2).mean(1)
    loss = squared_error.min(1)[0] + squared_error.min(2)[0]
    return loss.mean()


def outer(a, b=None):
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b


def per_sample_hungarian_loss(sample_np):
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
    return row_idx, col_idx


def hungarian_loss(predictions, targets, mask, pool):
    # predictions and targets shape :: (n, c, s)
    predictions = predictions[:,:mask,:]
    targets = targets[:,:mask,:]
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = (predictions - targets).pow(2).mean(1)

    squared_error_np = squared_error.detach().cpu().numpy()
    indices = pool.map(per_sample_hungarian_loss, squared_error_np)
    losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss


# GNN encoder to encoder node embeddings, and classifying which Gaussian Distribution the node will fall
def generate_gt_neighbor(neighbor_dict, node_embeddings, neighbor_num_list, in_dim):
    max_neighbor_num = max(neighbor_num_list)
    all_gt_neighbor_embeddings = []
    for i, embedding in enumerate(node_embeddings):
        neighbor_indexes = neighbor_dict[i]
        neighbor_embeddings = []
        for index in neighbor_indexes:
            neighbor_embeddings.append(node_embeddings[index].tolist())
        if len(neighbor_embeddings) < max_neighbor_num:
            for _ in range(max_neighbor_num - len(neighbor_embeddings)):
                neighbor_embeddings.append(torch.zeros(in_dim).tolist())
        all_gt_neighbor_embeddings.append(neighbor_embeddings)
    return all_gt_neighbor_embeddings


class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num, max_degree_num, GNN_name="GIN"):
        super(GNNStructEncoder, self).__init__()
        self.n_distribution = 5 # How many gaussian distribution should exist
        if GNN_name == "GIN":
            self.linear1 = MLP(layer_num, in_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(apply_func=self.linear1, aggregator_type='sum')
            self.linear2 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(apply_func=self.linear2, aggregator_type='sum')
        self.linear_classifier = MLP(1, hidden_dim, hidden_dim, out_dim)
        # Gaussian Means, and std
        self.gaussian_mean = nn.Parameter(torch.FloatTensor(self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim))
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim))
        self.m = torch.distributions.Normal(torch.zeros(self.n_distribution, hidden_dim), torch.ones(self.n_distribution, hidden_dim))
        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, max_degree_num, 4)
        self.degree_loss_func = FocalLoss(int(max_degree_num) + 1)
        self.pool = mp.Pool(4)

    def forward_encoder(self, g, h):
        # Apply graph convolution and activation
        l1 = F.relu(self.graphconv1(g, h))
        l2 = F.relu(self.graphconv2(g, l1))
        l3 = F.relu(self.graphconv2(g, l2))
        l4 = self.graphconv2(g, l3)
        l5 = self.graphconv2(g, F.relu(l4)) # 5 layers
        # with g.local_scope():
        #     g.ndata['h'] = h
        return l5, l4

    def neighbor_decoder(self, gij, ground_truth_degree_matrix, gt_neighbor_embeddings, neighbor_num_list, g, h):
        degree_logits = self.degree_decoding(gij)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix)
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        max_neighbor_num = max(neighbor_num_list)
        for i, embedding in enumerate(gij):
            zij = F.gumbel_softmax(self.linear_classifier(embedding))
            generated_neighbors = []
            for _ in range(max_neighbor_num):
                std_z = self.m.sample()
                var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
                var = F.dropout(var, 0.2)
                nhij = zij @ var
                generated_neighbors.append(nhij.tolist())
            generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
            target_neighbors = torch.unsqueeze(torch.FloatTensor(gt_neighbor_embeddings[i]), dim=0)
            h_loss += hungarian_loss(generated_neighbors, target_neighbors, neighbor_num_list[i], self.pool)
        loss = h_loss + degree_loss
        return loss, self.forward_encoder(g, h)[0]

    def degree_decoding(self, node_embeddings):
        degree_logits = self.degree_decoder(node_embeddings)
        return degree_logits

    def forward(self, g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, in_dim):
        gij, l4 = self.forward_encoder(g, h)
        gt_neighbor_embeddings = generate_gt_neighbor(neighbor_dict, l4, neighbor_num_list, in_dim)
        loss, hij = self.neighbor_decoder(gij, ground_truth_degree_matrix, gt_neighbor_embeddings, neighbor_num_list, g, h)
        return loss, hij