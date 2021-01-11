import sys
sys.path.append("..")
from data import build_graph
import seaborn as sb
import dgl
import networkx as nx
import torch
import torch.nn.functional as F
from src.model import GNNStructEncoder
import torch.nn as nn
from sklearn.manifold import TSNE
from bioinfokit.visuz import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from data import shapes


def cluster_graph(role_id, node_embeddings):
    nb_clust = len(np.unique(role_id))
    pca = PCA(n_components=5)
    trans_data = pca.fit_transform(StandardScaler().fit_transform(node_embeddings.detach()))
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
    plt.show()

def draw_pca(role_id, node_embeddings):
    pca = PCA(n_components=2)
    node_embedded = StandardScaler().fit_transform(node_embeddings.detach())
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

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1- Start by defining our favorite regular structure

    width_basis = 15
    nbTrials = 20

    ################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
    ## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type

    ### 1. Choose the basis (cycle, torus or chain)
    basis_type = "string"

    ### 2. Add the shapes
    n_shapes = 5  ## numbers of shapes to add
    # shape=["fan",6] ## shapes and their associated required parameters  (nb of edges for the star, etc)
    # shape=["star",6]
    list_shapes = [["house"]] * n_shapes

    ### 3. Give a name to the graph
    identifier = 'AA'  ## just a name to distinguish between different trials
    name_graph = 'houses' + identifier
    sb.set_style('white')

    ### 4. Pass all these parameters to the Graph Structure
    add_edges = 0
    G, communities, _, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                                             add_random_edges=add_edges, plot=True,
                                                             savefig=False)

    # G, role_id = shapes.barbel_graph(0, 8, 5, plot=True)
    # set color
    cmap = plt.get_cmap('hot')
    x_range = np.linspace(0, 0.8, len(np.unique(role_id)))
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
    node_color = [coloring[role_id[i]] for i in range(len(role_id))]
    print(coloring)
    g = dgl.from_networkx(G)
    g.to(device)
    one_hot_feature = F.one_hot(g.in_degrees())
    g.ndata['attr'] = one_hot_feature
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())
    epoches = 10
    temp_min = 0.6
    ANNEAL_RATE = 0.00001
    temp = 1
    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    in_dim = 6
    # Train in_dim, hidden_dim, out_dim, layer_num, max_degree_num
    GNNModel = GNNStructEncoder(in_dim, in_dim, 7, 2, in_dim)
    GNNModel.to(device)
    opt = torch.optim.Adam(GNNModel.parameters(), lr=5e-4, weight_decay=0.00003)
    for i in range(500):
        feats = g.ndata['attr'].float()
        # g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, in_dim, temp
        loss, node_embeddings = GNNModel(g, feats, g.in_degrees(), neighbor_dict, neighbor_num_list, in_dim, temp)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
        if i == 0:
            ## Draw everything
            node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.detach().numpy())
            cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="beforetrain_tsne")
            cluster_graph(role_id, node_embeddings)
            draw_pca(role_id, node_embeddings)
        opt.zero_grad()
        loss.backward()
        print(loss.item())
        opt.step()
    node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.detach().numpy())
    cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="aftertrain_tsne")
    cluster_graph(role_id, node_embeddings)
    draw_pca(role_id, node_embeddings)