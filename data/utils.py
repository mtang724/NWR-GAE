### Random tools useful for saveing stuff and manipulating pickle/numpy objects
import numpy as np
import pickle
import gzip
import re
import networkx as nx
import dgl
import torch


import logging
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
# import build_graph
import seaborn as sb

class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.2, shuffle=True):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[ : validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


def save_obj(obj, name, path, compress=False):
    # print path+name+ ".pkl"
    if compress is False:
        with open(path + name + ".pkl", 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(path + name + '.pklz','wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name,compressed=False):
    if compressed is False:
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with gzip.open(name,'rb') as f:
            return pickle.load(f)


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(l):
    '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        float regex comes from https://stackoverflow.com/a/12643073/190597
        '''
    t = np.array([ int(re.split(r"([a-zA-Z]*)([0-9]*)", c)[2]) for c in l  ])
    order = np.argsort(t)
    return [l[o] for o in order]


def saveNet2txt(G, colors=[], name="net", path="plots/"):
    '''saves graph to txt file (for Gephi plotting)
    INPUT:
    ========================================================================
    G:      nx graph
    colors: colors of the nodes
    name:   name of the file
    path:   path of the storing folder
    OUTPUT:
    ========================================================================
    2 files containing the edges and the nodes of the corresponding graph
    '''
    if len(colors) == 0:
        colors = range(nx.number_of_nodes(G))
    graph_list_rep = [["Id","color"]] + [[i,colors[i]]
                      for i in range(nx.number_of_nodes(G))]
    np.savetxt(path + name + "_nodes.txt", graph_list_rep, fmt='%s %s')
    edges = G.edges(data=False)
    edgeList = [["Source", "Target"]] + [[v[0], v[1]] for v in edges]
    np.savetxt(path + name + "_edges.txt", edgeList, fmt='%s %s')
    print ("saved network  edges and nodes to txt file (for Gephi vis)")
    return


def read_real_datasets(datasets):
    edge_path = "../datasets/{}/out1_graph_edges.txt".format(datasets)
    node_feature_path = "../datasets/{}/out1_node_feature_label.txt".format(datasets)
    with open(edge_path) as edge_file:
        edge_file_lines = edge_file.readlines()
        G = nx.parse_edgelist(edge_file_lines[1:], nodetype=int)
        g = dgl.from_networkx(G)
    with open(node_feature_path) as node_feature_file:
        node_lines = node_feature_file.readlines()[1:]
        feature_list = []
        labels = []
        max_len = 0
        for node_line in node_lines:
            node_id, feature, label = node_line.split("\t")
            labels.append(int(label))
            features = feature.split(",")
            max_len = max(len(features), max_len)
            feature_list.append([float(feature) for feature in features])
        feature_pad_list = []
        for features in feature_list:
            features += [0] * (max_len - len(features))
            feature_pad_list.append(features)
        feature_array = np.array(feature_pad_list)
        features = torch.from_numpy(feature_array)
        g.ndata['attr'] = features.float()
        labels = np.array(labels)
        labels = torch.FloatTensor(labels)
    return g, labels

def synthetic_graph_generator(width_basis=15, basis_type = "cycle", n_shapes = 5, shape_list=[[["house"]]], identifier = 'AA', add_edges = 0, plot = False):
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
                                                                   plot=plot, savefig=False)
    return G, role_id, identifier
