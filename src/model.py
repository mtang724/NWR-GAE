from src.layers import MLP, MLP_generator, PairNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv, GraphConv

from src.utils import FocalLoss
import scipy
import scipy.optimize
import torch.multiprocessing as mp
import time
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


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


def hungarian_loss(predictions, targets, mask, pool):
    # predictions and targets shape :: (n, c, s)
    predictions = predictions[:,:mask,:]
    targets = targets[:,:mask,:]
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = (predictions - targets).pow(2).mean(1)
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
    def __init__(self, node_num, in_dim, hidden_dim, out_dim, layer_num, sample_size, device, neighbor_num_list, GNN_name="GIN", norm_mode='PN', norm_scale=1):
        super(GNNStructEncoder, self).__init__()
        self.norm = PairNorm(norm_mode, norm_scale)
        self.n_distribution = 7 # How many gaussian distribution should exist
        self.out_dim = hidden_dim
        if GNN_name == "GIN":
            self.linear1 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(apply_func=self.linear1, aggregator_type='sum')
            self.linear2 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(apply_func=self.linear2, aggregator_type='sum')
            self.linear3 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv3 = GINConv(apply_func=self.linear3, aggregator_type='sum')
            self.linear4 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv4 = GINConv(apply_func=self.linear4, aggregator_type='sum')
        else:
            self.graphconv1 = GraphConv(hidden_dim, hidden_dim)
            self.graphconv2 = GraphConv(hidden_dim, hidden_dim)
            self.graphconv3 = GraphConv(hidden_dim, hidden_dim)
            self.graphconv4 = GraphConv(hidden_dim, hidden_dim)
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
        l1 =self.graphconv1(g, h)
        l1_norm = torch.relu(self.norm(l1))
        # l1 = F.normalize(l1, p=1, dim=1)
        # l2 = self.graphconv2(g, l1_norm)
        # l2_norm = torch.relu(self.norm(l2))
        # l2 = torch.relu(self.norm(self.graphconv2(g, l1)))
        # l2 = F.normalize(l2, p=1, dim=1)
        # l3 = self.graphconv3(g, l2_norm)
        # l3_norm = torch.relu(self.norm(l3))
        # l3 = torch.relu(self.norm(self.graphconv2(g, l2)))
        # l3 = F.normalize(l3, p=1, dim=1)
        # l4 = F.normalize(l4, p=1, dim=1)
        l2_norm = 0
        l3_norm = 0
        l5 = self.graphconv4(g, l1_norm) # 5 layers
        # l5 = F.normalize(l5, p=1, dim=1)
        # with g.local_scope():
        #     g.ndata['h'] = h
        return l5, l1_norm, l2_norm, l3_norm

    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes, 5)
            for index in sample_indexes:
                sampled_embeddings.append(gt_embeddings[index].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
        return sampled_embeddings_list


    def neighbor_decoder(self, gij, ground_truth_degree_matrix, neighbor_num_list, temp, g, h, neighbor_dict, device, test, l3, l2, l1):
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        # print(degree_logits)
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        max_neighbor_num = max(neighbor_num_list)
        mask_len = self.sample_size
        # layer 1
        loss_list = []
        total_sample_time = 0
        total_matching_time = 0
        for _ in range(3):
            local_index_loss = 0
            indexes = []
            for i1, embedding in enumerate(gij):
                indexes.append(i1)
            # print(len(indexes))
            #   here
            # sampled_embeddings_list = self.sample_neighbors(indexes, neighbor_dict, l2)
            # # print(sampled_embeddings_list)
            # for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            #     index = indexes[i]
            #     # zij = F.gumbel_softmax(self.linear_classifier(gij[index]), tau=temp)
            #     std_z = self.m.sample().to(device)
            #     var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
            #     var = F.dropout(var, 0.2)
            #     # nhij = zij @ var
            #     nhij = var
            #     # print (var.shape, nhij.shape)
            #     # print(nhij.shape)
            #     nhij = self.layer1_generator(nhij, device)
            #     generated_neighbors = nhij.tolist()
            #     generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
            #     target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0)
            #     new_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len, self.pool)
            #     local_index_loss += new_loss
            #   here
                # print("1", new_loss)
            # indexes2 = []
            # for i in indexes:
            #     indexes2 += neighbor_dict[i]
            # print (len(indexes2))
            sampled_embeddings_list = self.sample_neighbors(indexes, neighbor_dict, l1)
            for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
                index = indexes[i]
                # zij = F.gumbel_softmax(self.linear_classifier2(gij[index]), tau=temp)
                std_z = self.m.sample().to(device)
                var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
                var = F.dropout(var, 0.2)
                nhij = var
                # nhij = zij @ var
                nhij = self.layer2_generator(nhij, device)
                generated_neighbors = nhij.tolist()
                generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
                target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0)
                new_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len, self.pool)
                local_index_loss += new_loss
                # print("2", new_loss)
            # sampled_embeddings_list = self.sample_neighbors(indexes, neighbor_dict, l3)
            # for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            #     index = indexes[i]
            #     # zij = F.gumbel_softmax(self.linear_classifier2(gij[index]), tau=temp)
            #     std_z = self.m.sample().to(device)
            #     var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
            #     var = F.dropout(var, 0.2)
            #     nhij = var
            #     # nhij = zij @ var
            #     nhij = self.layer3_generator(nhij, device)
            #     generated_neighbors = nhij.tolist()
            #     generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
            #     target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0)
            #     new_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len, self.pool)
            #     local_index_loss += new_loss
            sampled_embeddings_list = self.sample_neighbors(indexes, neighbor_dict, h)
            for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
                index = indexes[i]
                # zij = F.gumbel_softmax(self.linear_classifier2(gij[index]), tau=temp)
                std_z = self.m_h.sample().to(device)
                var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
                var = F.dropout(var, 0.2)
                nhij = var
                # nhij = zij @ var
                nhij = self.layer4_generator(nhij, device)
                # print(nhij)
                # print(neighbor_embeddings1)
                generated_neighbors = nhij.tolist()
                generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
                target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0)
                new_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len, self.pool)
                local_index_loss += new_loss
            indexes3 = []
            # for i in indexes2:
            #     indexes3 += neighbor_dict[i]
            # print(len(indexes3))
            # sampled_embeddings_list = self.sample_neighbors(indexes, neighbor_dict, l1)
            # for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            #     index = indexes[i]
            #     zij = F.gumbel_softmax(self.linear_classifier3(gij[index]), tau=temp)
            #     std_z = self.m.sample().to(device)
            #     var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
            #     var = F.dropout(var, 0.2)
            #     nhij = zij @ var
            #     generated_neighbors = nhij.tolist()
            #     generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
            #     target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0)
            #     new_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len, self.pool)
            #     local_index_loss += new_loss
                # print("3", new_loss)
            # print (local_index_loss)
            loss_list.append(local_index_loss)
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        print (degree_loss)
        loss = h_loss + degree_loss * 10
            # layer 2
            # generated_neighbors = torch.squeeze(torch.FloatTensor(generated_neighbors), dim=0)
            # neighbor_indexes = neighbor_dict[i1]
            # new_neighbor_indexes = []
            # for index in new_index:
            #     new_neighbor_indexes.append(neighbor_indexes[index])
            # for i2, embedding in zip(new_neighbor_indexes, generated_neighbors):
            #     zij = F.gumbel_softmax(self.linear_classifier(embedding), tau=temp)
            #     std_z = self.m.sample().to(device)
            #     var = self.gaussian_mean + self.gaussian_log_sigma.exp() * std_z
            #     var = F.dropout(var, 0.2)
            #     nhij = zij @ var
            #     generated_neighbors = nhij.tolist()
            #     generated_neighbors = torch.unsqueeze(torch.FloatTensor(generated_neighbors), dim=0)
            #     target_neighbors = torch.unsqueeze(torch.FloatTensor(gt_neighbor_embeddings2[i2]), dim=0)
            #     new_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, neighbor_num_list[i2],
            #                                          self.pool)
        # print("sample time")
        # print(total_sample_time)
        # print("matching time")
        # print(total_matching_time)
        # print("degree loss")
        # print(degree_loss)
        # print("embedding loss")
        # print(h_loss)
        return loss, self.forward_encoder(g, h)[0]

    def degree_decoding(self, node_embeddings):
        degree_logits = self.degree_decoder(node_embeddings)
        return degree_logits

    def forward(self, g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, in_dim, temp, test, device):
        start_time = time.time()
        gij, l1, l2, l3 = self.forward_encoder(g, h)
        # print("encoder time:")
        # print(time.time() - start_time)
        # start_time = time.time()
        #gt_neighbor_embeddings1 = generate_gt_neighbor(neighbor_dict, l4, neighbor_num_list, in_dim)
        #gt_neighbor_embeddings2 = generate_gt_neighbor(neighbor_dict, l3, neighbor_num_list, in_dim)
        # print("generate neighborhood ground truth time:")
        # print(time.time() - start_time)
        # start_time = time.time()
        loss, hij = self.neighbor_decoder(gij, ground_truth_degree_matrix, neighbor_num_list, temp, g, h, neighbor_dict, device, test, l3, l2, l1)
        # print("decoder time:")
        # print(time.time() - start_time)
        return loss, hij