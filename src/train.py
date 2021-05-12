import sys
sys.path.append("..")
from data import build_graph, utils
import seaborn as sb
import dgl
import torch
import torch.nn.functional as F
from src.model import GNNStructEncoder
from bioinfokit.visuz import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sklearn as sk
from sklearn.manifold import TSNE
from tqdm import tqdm
import networkx as nx
from layers import MLP
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dgl.data import CitationGraphDataset
import statistics


class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def cluster_graph(role_id, node_embeddings):
    colors = role_id
    nb_clust = len(np.unique(role_id))
    pca = PCA(n_components=5)
    trans_data = pca.fit_transform(StandardScaler().fit_transform(node_embeddings.cpu().detach()))
    km = KMeans(n_clusters=nb_clust)
    km.fit(trans_data)
    labels_pred = km.labels_

    ######## Params for plotting
    cmapx = plt.get_cmap('rainbow')
    x = np.linspace(0, 1, nb_clust + 1)
    col = [cmapx(xx) for xx in x]
    markers = {0: '*', 1: '.', 2: ',', 3: 'o', 4: 'v', 5: '^', 6: '<', 7: '>', 8: 3, 9: 'd', 10: '+', 11: 'x',
               12: 'D', 13: '|', 14: '_', 15: 4, 16: 0, 17: 1, 18: 2, 19: 6, 20: 7}

    for c in np.unique(role_id):
        indc = [i for i, x in enumerate(role_id) if x == c]
        plt.scatter(trans_data[indc, 0], trans_data[indc, 1],
                    c=np.array(col)[list(np.array(labels_pred)[indc])],
                    marker=markers[c % len(markers)], s=300)

    labels = role_id
    for label, c, x, y in zip(labels, labels_pred, trans_data[:, 0], trans_data[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    # plt.show()
    return labels_pred, colors, trans_data, nb_clust


def unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust):
    ami = sk.metrics.adjusted_mutual_info_score(colors, labels_pred)
    sil = sk.metrics.silhouette_score(trans_data, labels_pred, metric='euclidean')
    ch = sk.metrics.calinski_harabasz_score(trans_data, labels_pred)
    hom = sk.metrics.homogeneity_score(colors, labels_pred)
    comp = sk.metrics.completeness_score(colors, labels_pred)
    #print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    #print(str(hom) + '\t' + str(comp) + '\t' + str(ami) + '\t' + str(nb_clust) + '\t' + str(ch) + '\t' + str(sil))
    return hom, comp, ami, nb_clust, ch, sil


def draw_pca(role_id, node_embeddings, coloring):
    pca = PCA(n_components=2)
    node_embedded = StandardScaler().fit_transform(node_embeddings.cpu().detach())
    principalComponents = pca.fit_transform(node_embedded)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    principalDf['target'] = role_id
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 PCA Components', fontsize=20)
    targets = np.unique(role_id)
    for target in zip(targets):
        color = coloring[target[0]]
        indicesToKeep = principalDf['target'] == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'],
                   principalDf.loc[indicesToKeep, 'principal component 2'],
                   s=50,
                   c=color)
    ax.legend(targets)
    ax.grid()
    plt.show()


def graph_generator(width_basis=15, basis_type = "cycle", n_shapes = 5, shape_list=[[["house"]]], identifier = 'AA', add_edges = 0):
    ################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
    ## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type
    ### 1. Choose the basis (cycle, torus or chain)
    ### 2. Add the shapes
    list_shapes = []
    for shape in shape_list:
        list_shapes += shape * n_shapes
    print(list_shapes)

    ### 3. Give a name to the graph
    name_graph = 'houses' + identifier
    sb.set_style('white')

    ### 4. Pass all these parameters to the Graph Structure
    G, communities, plugins, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                                                   add_random_edges=add_edges,
                                                                   plot=True, savefig=False)
    return G, role_id


def average(lst):
    return sum(lst) / len(lst)


def write_graph2edgelist(G, role_id, filename):
    nx.write_edgelist(G, "{}.edgelist".format(filename), data=False)
    with open("{}.roleid".format(filename), "w") as f:
        for id in role_id:
            f.write(str(id) + "\n")

def dimension_reduction(pca, embeddings):
    # pca = PCA(n_components=target_dim)
    low_dim_embeddings = pca.fit_transform(embeddings)
    return low_dim_embeddings


def set_pca( pca, embeddings):
    node_embedded = StandardScaler().fit_transform(embeddings)
    pca.fit(node_embedded)
    return pca

def train(g, feats, lr, epoch, device, encoder):
    # pca = PCA(n_components=320)
    # pca = set_pca(pca, feats.cpu().detach())
    # feats = torch.from_numpy(dimension_reduction(pca=pca, embeddings=feats.cpu().detach())).float().to(device)
    in_nodes, out_nodes = g.edges()
    node_num = g.nodes().shape[0]
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    temp_min = 0.2
    ANNEAL_RATE = 0.000001
    temp = 1
    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    # in_dim = max(neighbor_num_list) + 1
    in_dim = feats.shape[1]
    # node_num, in_dim, hidden_dim, out_dim, layer_num, sample_size, device, neighbor_num_list, GNN_name="GIN"
    sample_size = 10
    GNNModel = GNNStructEncoder(node_num, in_dim, in_dim, 100, 2, sample_size, device=device, neighbor_num_list=neighbor_num_list, GNN_name=encoder)
    GNNModel.to(device)
    degree_params = list(map(id, GNNModel.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params,
                         GNNModel.parameters())

    opt = torch.optim.Adam([{'params': base_params}, {'params': GNNModel.degree_decoder.parameters(), 'lr': 1e-2}],lr=lr, weight_decay=0.0003)
    for i in tqdm(range(epoch)):
        feats = feats.to(device)
        # g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, in_dim, temp
        loss, node_embeddings = GNNModel(g, feats, g.in_degrees(), neighbor_dict, neighbor_num_list, in_dim, temp,
                                         test=False, device=device)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
        # if i == 0:
        #     ## Draw everything
        #     node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.cpu().detach().numpy())
        #     cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="beforetrain_tsne")
        #     labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, node_embeddings)
        #     results = unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust)
        #     print(results)
        #     draw_pca(role_id, node_embeddings, coloring)
        opt.zero_grad()
        loss.backward()
        print(i, loss.item())
        opt.step()
    return node_embeddings.cpu().detach()


def train_synthetic_graphs(attributed = False):
    attributed = True
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    name_graph = 'barbell'
    sb.set_style('white')
    homs = []
    comps = []
    amis = []
    chs = []
    sils = []
    for test_iter in range(10):
        # G, role_id = graph_generator(width_basis=6,n_shapes = 2, shape_list=[[["house", 5]]], add_edges=0)
        G = nx.read_gpickle("../datasets/intro.gpickle")
        role_id = np.loadtxt("../datasets/intro.out")
        nb_clust = len(np.unique(role_id))
        print('nb of nodes in the graph: ', G.number_of_nodes())
        print('nb of edges in the graph: ', G.number_of_edges())
        # set color
        cmap = plt.get_cmap('hot')
        x_range = np.linspace(0, 0.8, len(np.unique(role_id)))
        coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
        node_color = [coloring[role_id[i]] for i in range(len(role_id))]
        print(G.nodes[13])
        g = dgl.from_networkx(G, node_attrs=["attr"])
        g = g.to(device)
        one_hot_feature = F.one_hot(g.in_degrees())
        print(one_hot_feature.shape)
        print(g.ndata['attr'].shape)
        if attributed:
            g.ndata['attr'] = g.ndata['attr'].float()
        else:
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
        in_dim = max(neighbor_num_list) + 1
        node_embeddings = train(g, g.ndata['attr'], lr=5e-3, epoch=200, device=device, encoder="GIN")
        node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.cpu().detach().numpy())
        # cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="aftertrain_tsne")
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

def train_real_datasets(dataset_str):
    gcn_setting = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CitationGraphDataset(dataset_str)
    g = dataset[0]
    g = g.to(device)
    node_features = g.ndata['feat']
    node_labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    valid_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)
    node_labels = g.ndata['label']
    # g, node_labels = utils.read_real_datasets("cornell")
    # g = g.to(device)
    # attr, feat
    acc = 0
    for i in range(10):
        node_embeddings = train(g, node_features, lr=5e-3, epoch=15, device=device, encoder="GCN")
        torch.save(node_embeddings.cpu().detach(), 'embeddings.pt')
        # node_embeddings = torch.load("embeddings.pt")
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
            for epoch in range(100):
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
            # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
            best = float('inf')
            for epoch in range(100):
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
                        torch.save(FNN.state_dict(), 'best_mlp.pkl')
                    print(str(epoch), correct / total)

            with torch.no_grad():
                FNN.load_state_dict(torch.load('best_mlp.pkl'))
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
            acc += (correct / total).item()
    print(acc/10)

def train_new_datasets(dataset_str):
    gcn_setting = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, labels = utils.read_real_datasets(dataset_str)
    g = g.to(device)
    node_features = g.ndata['attr']
    node_labels = labels
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)
    # g, node_labels = utils.read_real_datasets("cornell")
    # g = g.to(device)
    # attr, feat
    acc = []
    for i in range(3):
        node_embeddings = train(g, node_features, lr=5e-8, epoch=15, device=device, encoder="GCN")
        torch.save(node_embeddings.cpu().detach(), 'embeddings.pt')
        # node_embeddings = torch.load("embeddings.pt")
        input_dims = node_embeddings.shape
        print(input_dims[1])
        class_number = int(max(node_labels)) + 1
        FNN = MLP(num_layers=5, input_dim=input_dims[1], hidden_dim=input_dims[1]//2, output_dim=class_number).to(device)
        FNN = FNN.to(device)
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
    # train_synthetic_graphs()
    # train_real_datasets(dataset_str="cora")
    train_new_datasets(dataset_str="film")