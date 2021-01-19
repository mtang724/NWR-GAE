## Model

In this work, we propose an unsupervised structural node representation model which follows a novel design of a encoder (Graph Isomorphism Network (GIN) encoder) - decoder(neighborhood information decoder) framework. This model leverage the strong proximity perserving power of modern GNNs, and continuous self-refining its structural representations for each node by decoding and adjusting its neighborhood embedded information, each node presents the structural identity by learning a latent embedding vector.  # Need Pan 为什么这样一个framework 能使node embedding更好

Let's $G = (V, E)$ denote a input graph with node feature vector $X(v)$ and neighborhood sets $\mathcal N(v)$ for $v \in V$. We designed a GIN Encoder Network to encode the initial node feature and graph structure information for $G$.

### GIN Encoder Network

![image-20210118000550230](/Users/mtang/Library/Application Support/typora-user-images/image-20210118000550230.png)

**Figure 1. GIN Encoder Network Structure, encode node information for k layers**

The goal of our encoder network is to encoding the k-hops neighborhood information for each node as a untrained node embedding. The GIN encoder iteratively updates the representation of node by aggregating represenations of its neighbors with a multi-layer perceptron (MLP) function. This way, the k-layer encoded node embeddings can capture the structural information within its k-hop network neighborhood. The formal GIN updates node represenations can be written as

$$h_v^{(k)} = MLP^{(k)} ((1+\epsilon^{k}) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal N(v)} h_u^{k-1}$$

In our setting, the $\epsilon$ and MLP parameters are learnable, the initial (layer 0) node representation $h_v^0 = X(v)$. We takes the k layer node embeddings result as a input and other layers embeddings as neighborhood references for our decoder model.

### Neighborhood Information Decoder

![image-20210118024139340](/Users/mtang/Library/Application Support/typora-user-images/image-20210118024139340.png)

**Figure 2. Neighbor Decoder Network, consist of degree of neighbors decoder and embedding of neighbors decoder**

The decoder model consists of two parts, 

1) Node degree decoder, uses a L layer feed-forward neural network with ReLU activations to predict the degree number $d(v)$ of a given node $h_v^{(k)}$

$$d(v) = W_{l}^T(ReLU(W_{l-1}^T h_v^k + b_{l-1})) + b_l, \quad l \in 1..L$$

2) Neighbor embeddings decoder, samples neighborhood embeddings $SN_i(V_n^k)$ for $i \in 1..d(v)$ of a given nodes, then mapping and adjusting the embeddings according to the reference previous level node embeddings $h(\mathcal N(v)^{k-1})$

For the **Node degree decoder**, let $D(v)$ denotes the ground truth degree of a node v.  The decoder tries to predict the node's degree with a mean squared error loss.

$$L_{nd} = \sum_{v \in V} \frac{(d(v)-D(v))^2}{|V|}$$

To sample and decode the neighborhood embeddings, we introduce our neighbor embeddings decoder, which has 



$$ S\mathcal N(v_n)^k = \sum_{m=1}^{M} z_m \cdot gd_m$$

For each node v, $z_m$ follows a local multinomial distribution and serve as a categorical variable for mutimodal Gaussian distribution combination. We uses a simple L layer FNN + gumbel softmax network to asign the class probabilities $\pi_1 ... \pi_M$ to categorical variables $z_1...z_M$

$$X = W_{l}^T(ReLU(W_{l-1}^T h_v^k + b_{l-1})) + b_l, \quad l \in 1..L$$

$$\Pi = softmax(X), |\Pi| = M$$

$z_m = \frac{exp(\frac{log(\pi_i) + g_i}{\tau})}{\sum_{j=1}^M exp(\frac{log(\pi_i) + g_i}{\tau})}, \quad \pi_i, \pi_j \in \Pi, \quad for \quad i = 1..M$

Where the initial tempuature value is 1.0, gradually decreased to 0.3, and $g_1..g_M$ are i.i.d samples drawn from *Gumbel*(0, 1)

and there are M independent multivariate gaussian distributions $gd_m$ which belongs to the same set of distributions $\mathcal GD$

$$gd_m \sim \mathcal N(\mu_m, \sigma_m^2), \quad gd_m \in \mathcal GD$$

![image-20210119130243304](/Users/mtang/Library/Application Support/typora-user-images/image-20210119130243304.png)

**Figure 3. Set matching for sampled layer k neighbor embeddings and groud truth layer k-1 neighbor embeddings**

To consturct the neighborhood embeddings of a node $v$, we compares the sampled neighboorhood embeddings $S\mathcal N(v_n)^k$ with the ground truth upper level node embeddings $\mathcal N(h(v_n)^{k-1})$. Because both of the embedding sets have xxx permutation probabilities, so there involves a linear assignment problem, which can be solved by Hungarian Algorithm in $O(n^3)$ times. We uses a pairwise loss as assignment costs:

$$L_{hun}(S\mathcal N(v_n)^k, \mathcal N(h(v_n)^{k-1})) = min_{\pi \in \Pi} ||S\mathcal N(v_n)_i^k - \mathcal N_\pi(h(v_n)^{k-1})_i||$$

where $\Pi$ is the space of permutations, each $\pi$ matches up the element in  $S\mathcal N(v_n)^k$ to the closest element in $\mathcal N(h(v_n)^{k-1})$ 



