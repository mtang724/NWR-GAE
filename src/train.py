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
from data import shapes

if __name__ == '__main__':
    # 1- Start by defining our favorite regular structure

    width_basis = 15
    nbTrials = 20

    # EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE
    # REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type

    # 1. Choose the basis (cycle, torus or chain)
    basis_type = "string"

    # 2. Add the shapes
    nb_shapes = 5  # numbers of shapes to add
    # shape = ["fan",6] ## shapes and their associated required parameters  (nb of edges for the star, etc)
    # shape = ["star",6]
    list_shapes = [["house"]] * nb_shapes

    # 3. Give a name to the graph
    identifier = 'AA'  # just a name to distinguish between different trials
    name_graph = 'houses' + identifier
    sb.set_style('white')

    # 4. Pass all these parameters to the Graph Structure
    add_edges = 4  # nb of edges to add anywhere in the structure
    del_edges = 0

    G, communities, plugins, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                                       rdm_basis_plugins=False, add_random_edges=0,
                                                       plot=True, savefig=False)

    #G, role_id = shapes.barbel_graph(0, 8, 5, plot=True)
    # set color
    cmap = plt.get_cmap('hot')
    x_range = np.linspace(0, 0.8, len(np.unique(role_id)))
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
    node_color = [coloring[role_id[i]] for i in range(len(role_id))]
    print(coloring)
    g = dgl.from_networkx(G)
    one_hot_feature = F.one_hot(g.in_degrees())
    g.ndata['attr'] = one_hot_feature
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())
    epoches = 10
    in_dim = 6
    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    # Train in_dim, hidden_dim, out_dim, layer_num, max_degree_num
    GNNModel = GNNStructEncoder(6, 6, 5, 2, 6)
    opt = torch.optim.Adam(GNNModel.parameters())
    for i in range(1500):
        feats = g.ndata['attr'].float()
        loss, node_embeddings = GNNModel(g, feats, g.in_degrees(), neighbor_dict, neighbor_num_list, in_dim)
        if i == 0:
            node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.detach().numpy())
            cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="beforetrain_tsne")
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
        opt.zero_grad()
        loss.backward()
        print(loss.item())
        opt.step()
    node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.detach().numpy())
    cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="aftertrain_tsne")
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

