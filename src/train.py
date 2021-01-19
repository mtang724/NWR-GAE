import sys
sys.path.append("..")
from data import build_graph
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


def unsupervised_evaluate(colors, labels_pred, nb_clust):
    ami = sk.metrics.adjusted_mutual_info_score(colors, labels_pred)
    sil = sk.metrics.silhouette_score(trans_data, labels_pred, metric='euclidean')
    ch = sk.metrics.calinski_harabasz_score(trans_data, labels_pred)
    hom = sk.metrics.homogeneity_score(colors, labels_pred)
    comp = sk.metrics.completeness_score(colors, labels_pred)
    #print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    #print(str(hom) + '\t' + str(comp) + '\t' + str(ami) + '\t' + str(nb_clust) + '\t' + str(ch) + '\t' + str(sil))
    return hom, comp, ami, nb_clust, ch, sil


def draw_pca(role_id, node_embeddings):
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
    #plt.show()


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


def Average(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # house
    # G, role_id = graph_generator(width_basis=30, n_shapes = 10)
    # house perturbed
    G, role_id = graph_generator(width_basis=30, n_shapes=10, add_edges=8)
    # Varied
    # G, role_id = graph_generator(width_basis=40, n_shapes = 8,
    #                             shape_list=[[["fan", 6]], [["star", 10]], [["house"]]])
    # Varied Perturbed
    #G, role_id = graph_generator(width_basis=25,
    #                             shape_list=[[["fan", 6]], [["star", 10]], [["house"]]], add_edges=10)

    ################################### EXAMPLE TO BUILD A MORE COMPLICATED STRUCTURE ##########
    ######### Alternatively, to define a structure with different types of patterns, pass them as a list
    ######### In the following example, we have 3 fans (with param. 6), 3 stars on 4 nodes, and 3 house shapes
    # name_graph = 'regular'
    # from sklearn import preprocessing
    # basis_type = "cycle"
    #
    # width_basis = 25
    # add_edges = 10
    # list_shapes = [["fan", 6]] * 5 + [["star", 10]] * 5 + [["house"]] * 5
    # G, colors_shape, plugins, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
    #                                                    add_random_edges=add_edges, plot=True, savefig=False)
    # print (list_shapes)
    # house_graphs = []
    # house_role_ids = []
    # for _ in range(25):
    #    G, communities, plugins, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
    #                                                                   rdm_basis_plugins=False, add_random_edges=add_edges,
    #                                                                   plot=False, savefig=False)
    #    house_graphs.append(G)
    #    house_role_ids.append(role_id)

    # set color
    cmap = plt.get_cmap('hot')
    x_range = np.linspace(0, 0.8, len(np.unique(role_id)))
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
    node_color = [coloring[role_id[i]] for i in range(len(role_id))]
    print(coloring)
    g = dgl.from_networkx(G)
    g = g.to(device)
    one_hot_feature = F.one_hot(g.in_degrees())
    g.ndata['attr'] = one_hot_feature.float()
    print(g.ndata['attr'].shape)
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    temp_min = 0.3
    ANNEAL_RATE = 0.00001
    temp = 1
    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    in_dim = max(neighbor_num_list) + 1
    # Train in_dim, hidden_dim, out_dim, layer_num, max_degree_num
    homs = []
    comps = []
    amis = []
    chs = []
    sils = []
    for test_iter in range(1):
        GNNModel = GNNStructEncoder(in_dim, in_dim, 7, 2, in_dim, device=device)
        GNNModel.to(device)
        opt = torch.optim.Adam(GNNModel.parameters(), lr=5e-3, weight_decay=0.00003)
        for i in tqdm(range(100)):
            feats = g.ndata['attr']
            feats = feats.to(device)
            # g, h, ground_truth_degree_matrix, neighbor_dict, neighbor_num_list, in_dim, temp
            loss, node_embeddings = GNNModel(g, feats, g.in_degrees(), neighbor_dict, neighbor_num_list, in_dim, temp, test = False, device=device)
            if i % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
            if i == 0:
                ## Draw everything
                node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.cpu().detach().numpy())
                cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="beforetrain_tsne")
                labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, node_embeddings)
                results = unsupervised_evaluate(colors, labels_pred, nb_clust)
                print(results)
                draw_pca(role_id, node_embeddings)
            opt.zero_grad()
            loss.backward()
            print(i, loss.item())
            opt.step()
        node_embedded = TSNE(n_components=2).fit_transform(node_embeddings.cpu().detach().numpy())
        cluster.tsneplot(score=node_embedded, colorlist=role_id, figname="aftertrain_tsne")
        labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, node_embeddings)
        hom, comp, ami, nb_clust, ch, sil = unsupervised_evaluate(colors, labels_pred, nb_clust)
        homs.append(hom)
        comps.append(comp)
        amis.append(ami)
        chs.append(ch)
        sils.append(sil)
        print("test iter:", str(test_iter))
    print(homs)
    print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    print(str(Average(homs)), str(Average(comps)), str(Average(amis)), str(nb_clust), str(Average(chs)), str(Average(sils)))
    draw_pca(role_id, node_embeddings)