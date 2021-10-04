from src.layers import MLP, MLP_generator, PairNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv, GraphConv, SAGEConv

from src.utils import FocalLoss
import scipy
import scipy.optimize
import torch.multiprocessing as mp
import time
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import norm
import math
from scipy.optimize import linear_sum_assignment


# FNN with gumbal_softmax
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        # x = F.gumbel_softmax(x)
        return x


def chamfer_loss(predictions, targets, mask):
    if mask == 0:
        return 0
    predictions = predictions[:, :mask, :]
    targets = targets[:, :mask, :]
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


def hungarian_loss1(boxesA, boxesB, mask, maximize=True):
    # print(mask)
    boxesA = boxesA[:mask, :]
    boxesB = boxesB[:mask, :]
    n = max(len(boxesA), len(boxesB))
    cost_matrix = torch.zeros((n,n))
    # print(n)

    for i, boxA in enumerate(boxesA):
        for j, boxB in enumerate(boxesB):
            if boxA is None or boxB is None:
                cost_matrix[i,j] = int(not maximize)
            else:
                cost_matrix[i, j] = torch.norm(boxA - boxB)
    cost_matrix = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=maximize)
    return torch.mean(torch.FloatTensor(cost_matrix[row_ind, col_ind]))

def hungarian_loss(predictions, targets, mask, pool):
    # predictions and targets shape :: (n, c, s)
    predictions = predictions[:,:mask,:]
    targets = targets[:,:mask,:]
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = torch.sqrt((predictions - targets).pow(2).mean(1))
    squared_error_np = squared_error.detach().cpu().numpy()
    indices = pool.map(per_sample_hungarian_loss, squared_error_np)
    # print(indices)
    losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss, indices[0][1]


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
    def __init__(self, node_num, in_dim, hidden_dim, out_dim, layer_num, sample_size, device, neighbor_num_list, GNN_name="GIN", norm_mode="PN-SCS", norm_scale=20, lambda_loss=0.0001):
        super(GNNStructEncoder, self).__init__()
        self.norm = PairNorm(norm_mode, norm_scale)
        self.n_distribution = 7 # How many gaussian distribution should exist
        self.out_dim = hidden_dim
        self.lambda_loss = lambda_loss
        if GNN_name == "GIN":
            self.linear1 = MLP(layer_num, in_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(apply_func=self.linear1, aggregator_type='sum')
            self.linear2 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(apply_func=self.linear2, aggregator_type='sum')
            self.linear3 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv3 = GINConv(apply_func=self.linear3, aggregator_type='sum')
            self.linear4 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv4 = GINConv(apply_func=self.linear4, aggregator_type='sum')
        elif GNN_name == "GCN":
            self.graphconv1 = GraphConv(in_dim, hidden_dim)
            self.graphconv2 = GraphConv(hidden_dim, hidden_dim)
            self.graphconv3 = GraphConv(hidden_dim, hidden_dim)
            self.graphconv4 = GraphConv(hidden_dim, hidden_dim)
        else:
            self.graphconv1 = SAGEConv(in_dim, hidden_dim, aggregator_type='mean')
            self.graphconv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
            self.graphconv3 = SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
            self.graphconv4 = SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
        self.neighbor_num_list = neighbor_num_list
        self.linear_classifier = MLP(1, hidden_dim, hidden_dim, self.n_distribution)
        self.linear_classifier2 = MLP(1, hidden_dim, hidden_dim, self.n_distribution)
        self.linear_classifier3 = MLP(1, hidden_dim, hidden_dim, self.n_distribution)
        self.neighbor_generator = MLP_generator(hidden_dim, hidden_dim, sample_size).to(device)
        # Gaussian Means, and std
        # self.gaussian_mean = nn.Parameter(torch.FloatTensor(sample_size, self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        # self.gaussian_log_sigma = nn.Parameter(
        #     torch.FloatTensor(sample_size, self.n_distribution, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        # self.m = torch.distributions.Normal(torch.zeros(sample_size, self.n_distribution, hidden_dim), torch.ones(sample_size, self.n_distribution, hidden_dim))

        self.gaussian_mean = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim,
                                                                                     0.5 / hidden_dim)).to(device)
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim,
                                                                                     0.5 / hidden_dim)).to(device)
        self.m = torch.distributions.Normal(torch.zeros(sample_size, hidden_dim),
                                            torch.ones(sample_size, hidden_dim))

        self.m_h = torch.distributions.Normal(torch.zeros(sample_size, hidden_dim),
                                            50* torch.ones(sample_size, hidden_dim))

        # Before MLP Gaussian Means, and std

        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_m = torch.distributions.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim))

        self.mlp_mean = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_sigma = nn.Linear(hidden_dim, hidden_dim)

        self.layer1_generator = MLP_generator(hidden_dim, hidden_dim, sample_size)
        self.layer2_generator = MLP_generator(hidden_dim, hidden_dim, sample_size)
        self.layer3_generator = MLP_generator(hidden_dim, hidden_dim, sample_size)
        self.layer4_generator = MLP_generator(hidden_dim, hidden_dim, sample_size)
        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        # self.degree_loss_func = FocalLoss(int(max_degree_num) + 1)
        self.degree_loss_func = nn.MSELoss()
        self.pool = mp.Pool(1)
        self.in_dim = in_dim
        self.sample_size = sample_size

    def forward_encoder(self, g, h):
        # Apply graph convolution and activation
        # l1 = torch.relu(self.norm(self.graphconv1(g, h)))
        l1 = self.graphconv1(g, h)
        l1_norm = F.relu(self.norm(l1))
        # l2 = self.graphconv2(g, l1_norm)
        # l2_norm = torch.relu(self.norm(l2))
        # l2 = torch.relu(self.norm(self.graphconv2(g, l1)))
        # l2 = F.normalize(l2, p=1, dim=1)
        # l3_norm = torch.relu(self.norm(l3))
        # l3 = torch.relu(self.norm(self.graphconv2(g, l2)))
        l2_norm = 0
        l3_norm = 0
        l5 = self.graphconv4(g, l1_norm) # 5 layers
        # l5 = F.normalize(l5, p=1, dim=1)
        # with g.local_scope():
        #     g.ndata['h'] = h
        return l5, l2_norm, l3_norm, l1_norm

    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes, self.sample_size)
                mask_len = self.sample_size
            for index in sample_indexes:
                sampled_embeddings.append(gt_embeddings[index].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
            mark_len_list.append(mask_len)
        return sampled_embeddings_list, mark_len_list


    def neighbor_decoder(self, gij, ground_truth_degree_matrix, neighbor_num_list, temp, g, h, neighbor_dict, device, test, l3, l2, l1):
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        # degree_loss = 0
        # print(degree_logits)
        # degree_loss = torch.tensor(0.0, requires_grad=True)
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        # h_loss = torch.tensor(0.0, requires_grad=True)
        h_loss = 0
        max_neighbor_num = max(neighbor_num_list)
        mask_len1 = self.sample_size
        # layer 1
        loss_list = []
        total_sample_time = 0
        total_matching_time = 0
        for _ in range(4):
            local_index_loss = 0
            indexes = []
            for i1, embedding in enumerate(gij):
                indexes.append(i1)
            error_list = []
            sampled_embeddings_list, mark_len_list = self.sample_neighbors(indexes, neighbor_dict, l1)
            for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
                index = indexes[i]
                mask_len1 = mark_len_list[i]
                # zij = F.gumbel_softmax(self.linear_classifier2(gij[index]), tau=temp)
                mean = gij[index].repeat(self.sample_size, 1)
                mean = self.mlp_mean(mean)

                sigma = gij[index].repeat(self.sample_size, 1)
                sigma = self.mlp_sigma(sigma)

                std_z = self.m.sample().to(device)
                # var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
                var = mean + sigma.exp() * std_z
                # print(var)
                # var = F.dropout(var, 0.1)
                # nhij = var
                # nhij = zij @ var
                nhij = self.norm(self.layer4_generator(var, device))
                # print(nhij)
                generated_neighbors = nhij
                # print(generated_neighbors)
                sum_neighbor_norm = 0
                for indexi, generated_neighbor in enumerate(generated_neighbors):
                    sum_neighbor_norm += torch.norm(generated_neighbor) / math.sqrt(self.out_dim)
                    # if indexi == 0:
                    #     print(torch.FloatTensor(generated_neighbor))
                    # print(indexi)
                avg_neighbor_norm = sum_neighbor_norm / self.sample_size
                # hun_loss = hungarian_loss1(generated_neighbors, torch.FloatTensor(neighbor_embeddings1), mask_len1)
                # print("hun loss", hun_loss)
                generated_neighbors = torch.unsqueeze(generated_neighbors, dim=0)
                target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0)
                # print(generated_neighbors[0][0], target_neighbors[0][0])
                hun_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len1, self.pool)
                avg_hun_loss = hun_loss
                # new_loss = chamfer_loss(generated_neighbors, target_neighbors, mask_len)
                error_tuple = (avg_neighbor_norm, avg_hun_loss)
                local_index_loss += hun_loss
                error_list.append(error_tuple)
            loss_list.append(local_index_loss)
            with open("error_tuple.txt", "w") as f:
                for error_tuple in error_list:
                    f.write(str(error_tuple[0].item()) + "," + str(error_tuple[1].item()) + "\n")
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        print (degree_loss)
        # herer
        loss = self.lambda_loss * h_loss + degree_loss * 10
        return loss, self.forward_encoder(g, h)[0]

    def degree_decoding(self, node_embeddings):
        degree_logits = self.degree_decoder(node_embeddings)
        return degree_logits

    def forward(self, g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, in_dim, temp, test, device):
        gij, l2, l3, l1 = self.forward_encoder(g, h)
        loss, hij = self.neighbor_decoder(gij, ground_truth_degree_matrix, neighbor_num_list, temp, g, h, neighbor_dict, device, test, l3, l2, l1)
        return loss, hij