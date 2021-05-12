import utils
import networkx as nx
import numpy as np
from dgl.data import CitationGraphDataset

mode = "introduction"
if mode == "real":
    dataset_str = "wisconsin"
    g,label = utils.read_real_datasets(dataset_str)
    nxg = g.to_networkx()
    nx.write_edgelist(nxg, "../edgelists/realdatasets/{}.edgelist".format(dataset_str))
    label = np.array(label)
    np.savetxt('../edgelists/realdatasets/np_{}.txt'.format(dataset_str), label)
    nx.write_gml(nxg, "../edgelists/realdatasets/{}.gml".format(dataset_str))

if mode == "citation":
    dataset_str = "pubmed"
    dataset = CitationGraphDataset(dataset_str)
    g = dataset[0]
    label = g.ndata['label']
    nxg = g.to_networkx()
    nx.write_edgelist(nxg, "../edgelists/citation/{}.edgelist".format(dataset_str))
    label = np.array(label)
    np.savetxt('../edgelists/citation/np_{}.txt'.format(dataset_str), label)
    nx.write_gml(nxg, "../edgelists/citation/{}.gml".format(dataset_str))

if mode == "synthetic":
    for i in range(10):
        G, role_id, identifier = utils.synthetic_graph_generator(width_basis=15, identifier="cycle_test", shape_list=[[["house"]]] , add_edges=0, plot=True)
        nxg = G
        nx.write_edgelist(nxg, "../edgelists/synthetic/{}.edgelist".format(identifier + str(i)))
        nx.write_gml(G, "../edgelists/synthetic/{}.gml".format(identifier + str(i)))
        label = np.array(role_id)
        np.savetxt('../edgelists/synthetic/np_{}.txt'.format(identifier + str(i)), label)

if mode == "introduction":
    G, role_id, identifier = utils.synthetic_graph_generator(width_basis=2, identifier="cycle_test", n_shapes=4,
                                                             shape_list=[[["house"]]], add_edges=0, plot=True, basis_type="string")

