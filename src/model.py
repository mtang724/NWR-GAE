from layers import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import GINConv
import dgl

from utils import FocalLoss, HungarianLoss

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


# GNN encoder to encoder node embeddings, and classifying which Gaussian Distribution the node will fall
class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num, max_degree_num, GNN_name="GIN"):
        super(GNNStructEncoder, self).__init__()
        self.n_distribution = out_dim # How many gaussian distribution should exist
        if GNN_name == "GIN":
            self.linear = MLP(layer_num, in_dim, hidden_dim, hidden_dim)
            self.graphconv = GINConv(apply_func=self.linear, aggregator_type='sum')
        self.linear_classifier = MLP(1, hidden_dim, hidden_dim, out_dim)
        # Relation Means, and std
        self.relations_mean = nn.Parameter(torch.FloatTensor(self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim))
        self.relations_log_sigma = nn.Parameter(
            torch.FloatTensor(self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim))
        self.m = torch.distributions.Normal(torch.zeros(self.n_distribution, hidden_dim), torch.ones(self.n_distribution, hidden_dim))
        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, max_degree_num, 4)
        self.degree_loss_func = FocalLoss(int(max_degree_num) + 1)

    def forward_encoder(self, g, h):
        # Apply graph convolution and activation
        node_embeddings = self.graphconv(g, h)
        self.node_embeddings = node_embeddings
        h = F.relu(node_embeddings)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout. We don't need it in this case
            # hg = dgl.sum_nodes(g, 'h')
        return h

    def neighbor_decoder(self, gij, ground_truth_degree_matrix, gt_neighbor_embeddings):
        hij = self.node_embeddings
        degree_logits = self.degree_decoding(hij)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix)
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        zij = F.gumbel_softmax(self.linear_classifier(gij))
        std_z = self.m.sample()
        var = self.relations_mean + self.relations_log_sigma.exp() * std_z
        # var = F.dropout(var, self.dropout, )
        nhij = zij @ var
        loss = self.hungarian_loss(nhij, gt_neighbor_embeddings) + degree_loss
        return loss, hij

    def degree_decoding(self, node_embeddings):
        degree_logits = self.degree_decoder(node_embeddings)
        return degree_logits

    def forward(self, g, h, ground_truth_degree_matrix, gt_neighbor_embeddings):
        gij = self.forward_encoder(g, h)
        loss, hij = self.neighbor_decoder(self, gij, ground_truth_degree_matrix, gt_neighbor_embeddings)

    def hungarian_loss(self, nhji, gt_neighbor_embeddings):
        loss = 0
        return loss