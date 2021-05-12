import sys
sys.path.append("..")
from data import build_graph, utils

g, labels = utils.read_real_datasets("wisconsin")
feature = g.ndata['attr']
print(feature.shape)