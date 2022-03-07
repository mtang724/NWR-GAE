import sys
sys.path.append("..")
from data import utils
import seaborn as sb
import dgl
import torch
from src.model import GNNStructEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
from layers import MLP
import torch.nn.functional as F
from dgl.data import CitationGraphDataset
import statistics
import argparse
import random
from utils import NodeClassificationDataset, cluster_graph, unsupervised_evaluate, draw_pca, graph_generator, average

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# Training
def train(g, feats, lr, epoch, device, encoder, lambda_loss1, lambda_loss2, hidden_dim, sample_size=10):
    '''
     Main training function
     INPUT:
     -----------------------
     g    :    graph
     feats     :   graph features
     lr    :    learning rate
     epoch     :    number of training epoch
     device     :   CPU or GPU
     encoder    :    GCN or GIN or GraphSAGE
     lambda_loss    :   Trade-off between degree loss and neighborhood reconstruction loss
     hidden_dim     :   latent variable dimension
    '''
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    in_dim = feats.shape[1]
    GNNModel = GNNStructEncoder(in_dim, hidden_dim, 2, sample_size, device=device, neighbor_num_list=neighbor_num_list, GNN_name=encoder, lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2)
    GNNModel.to(device)
    degree_params = list(map(id, GNNModel.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params,
                         GNNModel.parameters())

    opt = torch.optim.Adam([{'params': base_params}, {'params': GNNModel.degree_decoder.parameters(), 'lr': 1e-2}],lr=lr, weight_decay=0.0003)
    for i in tqdm(range(epoch)):
        feats = feats.to(device)
        # g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, in_dim, temp
        loss, node_embeddings = GNNModel(g, feats, g.in_degrees(), neighbor_dict, device=device)
        opt.zero_grad()
        loss.backward()
        print(i, loss.item())
        opt.step()
    return node_embeddings.cpu().detach(), loss.item()


def train_synthetic_graphs(attributed = False):
    # attributed = True
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    name_graph = 'barbell'
    sb.set_style('white')
    homs = []
    comps = []
    amis = []
    chs = []
    sils = []
    for test_iter in range(10):
        # generate synthetic graph from GraphWave graph generator
        G, role_id = graph_generator(width_basis=6,n_shapes = 2, shape_list=[[["house", 5]]], add_edges=0)
        print('nb of nodes in the graph: ', G.number_of_nodes())
        print('nb of edges in the graph: ', G.number_of_edges())
        # set color
        cmap = plt.get_cmap('inferno')
        x_range = np.linspace(0, 0.9, len(np.unique(role_id)))
        coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
        if attributed:
            g = dgl.from_networkx(G, node_attrs=["attr"])
            g = g.to(device)
            g.ndata['attr'] = g.ndata['attr'].float()
        else:
            g = dgl.from_networkx(G)
            one_hot_feature = F.one_hot(g.in_degrees())
            g.ndata['attr'] = one_hot_feature.float()

        in_nodes, out_nodes = g.edges()
        neighbor_dict = {}
        for in_node, out_node in zip(in_nodes, out_nodes):
            if in_node.item() not in neighbor_dict:
                neighbor_dict[in_node.item()] = []
            neighbor_dict[in_node.item()].append(out_node.item())
        neighbor_num_list = []
        for i in neighbor_dict:
            neighbor_num_list.append(len(neighbor_dict[i]))
        node_embeddings, _ = train(g, g.ndata['attr'], lr=5e-3, epoch=100, device=device, encoder="SAGE", lambda_loss1=1e-1, lambda_loss2=1, hidden_dim=6)
        node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.cpu().detach().numpy())
        labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, node_embeddings)
        hom, comp, ami, nb_clust, ch, sil = unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust)
        print(hom, comp, ami, nb_clust, ch, sil)
        homs.append(hom)
        comps.append(comp)
        amis.append(ami)
        chs.append(ch)
        sils.append(sil)
        print("test iter:", str(test_iter))
        draw_pca(role_id, node_embeddings, coloring)
    print(homs)
    print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    print(str(average(homs)), str(average(comps)), str(average(amis)), str(nb_clust), str(average(chs)),
          str(average(sils)))


def evaluate(model, embeddings, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_real_datasets(dataset_str, epoch_num = 10, lr = 5e-6, encoder = "GCN", lambda_loss1=1e-4, lambda_loss2=1, sample_size=8, hidden_dim=None):
    gcn_setting = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CitationGraphDataset(dataset_str)
    g = dataset[0]
    g = g.to(device)
    node_features = g.ndata['feat']
    train_mask = g.ndata['train_mask']
    valid_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    node_labels = g.ndata['label']
    # attr, feat
    if hidden_dim == None:
        hidden_dim = node_features.shape[1]
    else:
        hidden_dim = hidden_dim
    acc = []
    for index in range(5):
        node_embeddings, _ = train(g, node_features, lr=lr, epoch=epoch_num, device=device, encoder=encoder, lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2, hidden_dim=hidden_dim, sample_size=sample_size)
        input_dims = node_embeddings.shape
        print(input_dims[1])
        class_number = int(max(node_labels)) + 1
        FNN = MLP(num_layers=5, input_dim=input_dims[1], hidden_dim=input_dims[1]//2, output_dim=class_number).to(device)
        FNN = FNN.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(FNN.parameters())
        if gcn_setting:
            inputs = node_embeddings
            inputs = inputs.to(device)
            for epoch in range(10):
                FNN.train()
                # forward propagation by using all nodes
                logits = FNN(inputs)
                # compute loss
                loss = criterion(logits[train_mask], node_labels[train_mask])
                # compute validation accuracy
                acc = evaluate(FNN, inputs, node_labels, valid_mask)
                print(acc)
                # backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
            acc = evaluate(FNN, inputs, node_labels, test_mask)
            print(acc)
        else:
            dataset = NodeClassificationDataset(node_embeddings, node_labels)
            split = utils.DataSplit(dataset, shuffle=True)
            train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
            best = float('inf')
            for epoch in range(50):
                for i, data in enumerate(train_loader, 0):
                    # data = data.to(device)
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    y_pred = FNN(inputs)
                    loss = criterion(y_pred, labels)
                    print(epoch, i, loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for data in val_loader:
                            inputs, labels = data
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = FNN(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            loss = criterion(outputs, labels)
                            total += labels.size(0)
                            correct += torch.sum(predicted == labels)
                    if loss < best:
                        best = loss
                        torch.save(FNN.state_dict(), 'best_mlp2.pkl')
                    print(str(epoch), correct / total)

            with torch.no_grad():
                FNN.load_state_dict(torch.load('best_mlp2.pkl'))
                correct = 0
                total = 0
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = FNN(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels)
            print((correct / total).item())
            acc.append((correct / total).item())


def train_new_datasets(dataset_str, epoch_num = 10, lr = 5e-6, encoder = "GCN", lambda_loss1=1e-4, lambda_loss2=1, sample_size=10, hidden_dim=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, labels = utils.read_real_datasets(dataset_str)
    g = g.to(device)
    node_features = g.ndata['attr']
    node_labels = labels
    # attr, feat
    if hidden_dim == None:
        hidden_dim = node_features.shape[1]
    else:
        hidden_dim = hidden_dim
    acc = []
    for index in range(5):
        node_embeddings, loss = train(g, node_features, lr=lr, epoch=epoch_num, device=device, encoder=encoder, lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2, sample_size=sample_size, hidden_dim=hidden_dim)
        input_dims = node_embeddings.shape
        print(input_dims[1])
        class_number = int(max(node_labels)) + 1
        FNN = MLP(num_layers=4, input_dim=input_dims[1], hidden_dim=input_dims[1]//2, output_dim=class_number).to(device)
        FNN = FNN.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(FNN.parameters())
        dataset = NodeClassificationDataset(node_embeddings, node_labels)
        split = utils.DataSplit(dataset, shuffle=True)
        train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
        # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        best = -float('inf')
        for epoch in range(50):
            for i, data in enumerate(train_loader, 0):
                # data = data.to(device)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_pred = FNN(inputs)
                loss = criterion(y_pred, labels)
                # train_loss = loss
                print(epoch, i, loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = FNN(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        total += labels.size(0)
                        correct += torch.sum(predicted == labels)
                if correct / total > best:
                    best = correct / total
                    torch.save(FNN.state_dict(), 'best_mlp_{}.pkl'.format(index))
                print(str(epoch), correct / total)
        with torch.no_grad():
            FNN.load_state_dict(torch.load('best_mlp_{}.pkl'.format(index)))
            correct = 0
            total = 0
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = FNN(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += torch.sum(predicted == labels)
        print((correct / total).item())
        acc.append((correct / total).item())
    print("mean:")
    print(statistics.mean(acc))
    print("std:")
    print(statistics.stdev(acc))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default="texas")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lambda_loss1', type=float, default=1e-2)
    parser.add_argument('--lambda_loss2', type=float, default=1e-2)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--dimension', type=int, default=1500)
    parser.add_argument('--identify', type=str, default="sample")
    parser.add_argument('--dataset_type', type=str, default="real")

    args = parser.parse_args()
    # train_synthetic_graphs()™
    if args.dataset_type == "real":
        dataset_str = args.dataset
        if dataset_str == "cora" or dataset_str == "citeseer" or dataset_str == "pubmed":
            train_real_datasets(dataset_str=dataset_str, lr=args.lr, epoch_num=args.epoch_num, lambda_loss1=args.lambda_loss1, lambda_loss2=args.lambda_loss2, encoder="GCN", sample_size=args.sample_size, hidden_dim=args.dimension)
        else:
            train_new_datasets(dataset_str=dataset_str, lr=args.lr, epoch_num=args.epoch_num, lambda_loss1=args.lambda_loss1, lambda_loss2=args.lambda_loss2, encoder="GCN", sample_size=args.sample_size, hidden_dim=args.dimension)
    else:
        train_synthetic_graphs()
