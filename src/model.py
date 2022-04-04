from src.layers import MLP, MLP_generator, PairNorm, FNN
from src.utils import hungarian_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv, GraphConv, SAGEConv
import torch.multiprocessing as mp
import random
import math


# generate ground truth neighbors Hv
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


# Main Autoencoder structure here
class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num, sample_size, device, neighbor_num_list, GNN_name="GIN", norm_mode="PN-SCS", norm_scale=20, lambda_loss1=0.0001, lambda_loss2=1):
        '''
         Main Autoencoder structure
         INPUT:
         -----------------------
         in_dim    :    input graph feature dimension
         hidden_dim     :   latent variable feature dimension
         layer_num    :    GIN encoder, number of MLP layer
         sample_size     :    number of neighbors sampled
         device     :   CPU or GPU
         neighbor_num_list    :    number of neighbors for a specific node
         norm   :   Pair Norm from https://openreview.net/forum?id=rkecl1rtwB
         lambda_loss    :   Trade-off between degree loss and neighborhood reconstruction loss
        '''
        super(GNNStructEncoder, self).__init__()
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        # GNN Encoder
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
        self.neighbor_generator = MLP_generator(hidden_dim, hidden_dim, sample_size).to(device)

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
        self.feature_decoder = FNN(hidden_dim, hidden_dim, in_dim, 3)
        # self.degree_loss_func = FocalLoss(int(max_degree_num) + 1)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.pool = mp.Pool(4)
        self.in_dim = in_dim
        self.sample_size = sample_size
        self.init_projection = FNN(in_dim, hidden_dim, hidden_dim, 1)

    def forward_encoder(self, g, h):
        # K-layer Encoder
        # Apply graph convolution and activation, pair-norm to avoid trivial solution
        h0 = h
        l1 = self.graphconv1(g, h0)
        l1_norm = torch.relu(self.norm(l1))
        l2 = self.graphconv2(g, l1_norm)
        l2_norm = torch.relu(self.norm(l2))
        l3 = self.graphconv3(g, l2)
        l3_norm = torch.relu(l3)
        l4 = self.graphconv4(g, l1_norm) # 5 layers
        return l4, l3_norm, l2_norm, l1_norm, h0

    # Sample neighbors from neighbor set, if the length of neighbor set less than sample size, then do the padding.
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

    def reconstruction_neighbors(self, FNN_generator, neighbor_indexes, neighbor_dict, from_layer, to_layer, device):
        '''
         Reconstruction Neighbors
         INPUT:
         -----------------------
         FNN_generator    :    FNN decoder
         neighbor_indexes     :   new neighbor indexes after hungarian matching
         neighbor_dict    :    specific neighbors a node have
         from_layer     :    from which layer K
         to_layer     :   decode to which layer K-1
         device    :    CPU or GPU
         OUTPUT:
         -----------------------
         loss   :   reconstruction loss
         new index    :   new indexes after hungarian matching
        '''
        local_index_loss = 0
        sampled_embeddings_list, mark_len_list = self.sample_neighbors(neighbor_indexes, neighbor_dict, to_layer)
        for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            # Generating h^k_v, reparameterization trick
            index = neighbor_indexes[i]
            mask_len1 = mark_len_list[i]
            mean = from_layer[index].repeat(self.sample_size, 1)
            mean = self.mlp_mean(mean)
            sigma = from_layer[index].repeat(self.sample_size, 1)
            sigma = self.mlp_sigma(sigma)
            std_z = self.m.sample().to(device)
            var = mean + sigma.exp() * std_z
            nhij = FNN_generator(var, device)
            generated_neighbors = nhij
            # Caculate 2-Wasserstein distance
            sum_neighbor_norm = 0
            for indexi, generated_neighbor in enumerate(generated_neighbors):
                sum_neighbor_norm += torch.norm(generated_neighbor) / math.sqrt(self.out_dim)
            generated_neighbors = torch.unsqueeze(generated_neighbors, dim=0).to(device)
            target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0).to(device)
            hun_loss, new_index = hungarian_loss(generated_neighbors, target_neighbors, mask_len1, self.pool)
            local_index_loss += hun_loss
            return local_index_loss, new_index

    def neighbor_decoder(self, gij, ground_truth_degree_matrix, g, h0, neighbor_dict, device, l3, l2, l1, h):
        '''
         Neighborhood information decoder
         INPUT:
         -----------------------
         gij    :    encoder output
         ground_truth_degree_matrix     :   ground truth degree information for each node as a matrix
         g    :    graph
         h     :    graph features
         neighbor_dict     :   specific neighbors a node have
         device    :    CPU or GPU
         l3, l2, l1   :   layer K encoding generated by encoder
         OUTPUT:
         -----------------------
         loss   :   degree loss + reconstruction loss
         latent variable    :   encoder output
        '''
        # Degree decoder below:
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        feature_loss = 0
        # layer 1
        loss_list = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for _ in range(3):
            local_index_loss_sum = 0
            indexes = []
            feature_losses = self.feature_loss_func(h0, self.feature_decoder(gij))
            feature_loss_list.append(feature_losses)
            for i1, embedding in enumerate(gij):
                indexes.append(i1)
            # Reconstruct neighbors from layer 4 -> 3 -> 2 -> 1
            local_index_loss, new_index = self.reconstruction_neighbors(self.layer4_generator, indexes, neighbor_dict, gij, l3, device)
            local_index_loss_sum += local_index_loss
            local_index_loss, new_index = self.reconstruction_neighbors(self.layer3_generator, new_index, neighbor_dict, l3, l2, device)
            local_index_loss_sum += local_index_loss
            local_index_loss, new_index = self.reconstruction_neighbors(self.layer2_generator, new_index, neighbor_dict, l2, l1, device)
            local_index_loss_sum += local_index_loss
            loss_list.append(local_index_loss_sum)
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        feature_loss_list = torch.stack(feature_loss_list)
        feature_loss += torch.mean(feature_loss_list)
        loss = self.lambda_loss1 * h_loss + degree_loss * 10 + self.lambda_loss2 * feature_loss
        return loss, self.forward_encoder(g, h)[0]

    def degree_decoding(self, node_embeddings):
        degree_logits = F.relu(self.degree_decoder(node_embeddings))
        return degree_logits

    def forward(self, g, h, ground_truth_degree_matrix, neighbor_dict, device):
        # Generate 1, .., k-1 layer GNN encodings
        gij, l2, l3, l1, h0 = self.forward_encoder(g, h)
        # Decoding and generating the latent representation by decoder, loss = degree_loss +
        loss, hij = self.neighbor_decoder(gij, ground_truth_degree_matrix, g, h0, neighbor_dict, device, l3, l2, l1, h)
        return loss, hij
