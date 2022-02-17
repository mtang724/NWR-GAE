import sys
sys.path.append("..")
from data import build_graph, utils
import seaborn as sb
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import scipy
import scipy.optimize
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn as sk
import networkx as nx

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        # P = F.softmax(inputs)
        P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class HungarianLoss():
    def __init__(self, predictions, targets, mask, pool):
        self.predictions = predictions
        self.targets = targets
        self.mask = mask
        self.pool = pool

    def outer(self, a, b=None):
        if b is None:
            b = a
        size_a = tuple(a.size()) + (b.size()[-1],)
        size_b = tuple(b.size()) + (a.size()[-1],)
        a = a.unsqueeze(dim=-1).expand(*size_a)
        b = b.unsqueeze(dim=-2).expand(*size_b)
        return a, b

    def per_sample_hungarian_loss(self, sample_np):
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
        return row_idx, col_idx

    def hungarian_loss(self):
        # predictions and targets shape :: (n, c, s)
        predictions = self.predictions[:self.mask,:]
        targets = self.targets[:self.mask,:]
        predictions = predictions.permute(0, 2, 1)
        targets = targets.permute(0, 2, 1)
        predictions, targets = self.outer(predictions, targets)
        # squared_error shape :: (n, s, s)
        squared_error = (predictions - targets).pow(2).mean(1)

        squared_error_np = squared_error.detach().cpu().numpy()
        indices = self.pool.map(self.per_sample_hungarian_loss, squared_error_np)
        losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
        total_loss = torch.mean(torch.stack(list(losses)))
        return total_loss


# Chamfer Loss, a more native and effecient loss calculation
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
    sample_np[sample_np == np.inf] = 0
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
    return row_idx, col_idx


## Two implementation for hungarian_loss (hungarian matching)
def hungarian_loss1(boxesA, boxesB, mask, maximize=True):
    # print(mask)
    boxesA = boxesA[:mask, :]
    boxesB = boxesB[:mask, :]
    n = max(len(boxesA), len(boxesB))
    cost_matrix = torch.zeros((n,n))
    # print(n)

    for i, boxA in enumerate(boxesA):
        for j, boxB in enumerate(boxesB):
            if boxA is None or boxB is None:
                cost_matrix[i,j] = int(not maximize)
            else:
                cost_matrix[i, j] = torch.norm(boxA - boxB)
    cost_matrix = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=maximize)
    return torch.mean(torch.FloatTensor(cost_matrix[row_ind, col_ind]))


def hungarian_loss(predictions, targets, mask, pool):
    # predictions and targets shape :: (n, c, s)
    predictions = predictions[:,:mask,:]
    targets = targets[:,:mask,:]
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = torch.sqrt((predictions - targets).pow(2).mean(1))
    squared_error_np = squared_error.detach().cpu().numpy()
    indices = pool.map(per_sample_hungarian_loss, squared_error_np)
    # print(indices)
    losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss, indices[0][1]


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

def set_pca( pca, embeddings):
    node_embedded = StandardScaler().fit_transform(embeddings)
    pca.fit(node_embedded)
    return pca
