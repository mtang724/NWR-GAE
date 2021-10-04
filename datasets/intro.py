import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_networkx(graph, role_labels):
    cmap = plt.get_cmap('inferno')
    x_range = np.linspace(0, 0.9, len(np.unique(role_labels)))
    print(x_range)
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_labels))}
    print(coloring)
    node_color = [coloring[role_labels[i]] for i in range(len(role_labels))]
    print(node_color)
    plt.figure()
    nx.draw_networkx(graph, pos=nx.layout.fruchterman_reingold_layout(graph),
                     node_color=node_color, cmap='inferno')
    plt.show()
    return

G = nx.Graph()
G.add_nodes_from(range(20))
# G.add_edges_from([(0, 1), (0, 6), (0, 7), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5), (6, 7), (6, 8),
#                   (6, 9), (7, 8), (7, 9), (8, 9), (10, 6), (10, 7), (10, 8), (10, 9), (11, 2), (11, 3), (11, 4), (11, 5),
#                   (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (13, 2), (13, 3), (13, 4), (13, 5), (13, 11),
#                   (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 12), (15, 2), (15, 3), (15, 4), (15, 5), (15, 11), (15, 13),
#                   (16, 6), (16, 7), (16, 8), (16, 9), (16, 10), (16, 12), (16, 14), (17, 2), (17, 3), (17, 4), (17, 5), (17, 11), (17, 13), (17, 15)])
# G.add_edges_from([(0, 1), (0, 6), (0, 7), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5), (6, 7), (6, 8),
#                   (6, 9), (7, 8), (7, 9), (8, 9)])
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (0, 5), (0, 6), (2, 4), (2, 5), (2, 6), (4, 5), (5, 6),
                  (1, 7), (1, 9), (3, 7), (3, 8), (3, 9), (7, 8), (4, 6), (7, 9), (8, 9), (1, 8)]

# G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (0, 5), (0, 6), (2, 4), (2, 5), (2, 6), (4, 5), (5, 6),
                  #(1, 7), (1, 9), (3, 7), (3, 8), (3, 9), (7, 8), (4, 6), (7, 9), (8, 9), (1, 8)])
for i in range(1, 11):
    node_id = 9 + i
    print(node_id)
    if node_id % 2 == 0:
        edges.append((0, node_id))
        edges.append((2, node_id))
        edges.append((4, node_id))
        edges.append((5, node_id))
        edges.append((6, node_id))
        for j in range(i):
            neighbor_id = 9 + j
            if neighbor_id % 2 == 0:
                edges.append((neighbor_id, node_id))
    else:
        edges.append((1, node_id))
        edges.append((3, node_id))
        edges.append((7, node_id))
        edges.append((8, node_id))
        edges.append((9, node_id))
        for j in range(i):
            neighbor_id = 9 + j
            if neighbor_id % 2 == 1:
                edges.append((neighbor_id, node_id))
G.add_edges_from(edges)
nx.draw(G)
plt.show()
#
# attrs = {0: {"role": 0, "attr": [1, 1, 1, 1, 1, 3]}, 1: {"role": 1, "attr": [100, 100, 100, 100, 100, 3]},
#          2: {"role": 2, "attr": [100, 100, 100, 100, 100, 8]}, 3: {"role": 2, "attr": [100, 100, 100, 100, 100, 8]},
#          4: {"role": 6, "attr": [1, 1, 1, 1, 1, 7]}, 5: {"role": 3, "attr": [100, 100, 100, 100, 100, 7]},
#          6: {"role": 4, "attr": [1, 1, 1, 1, 1, 8]}, 7: {"role": 4, "attr": [100, 100, 100, 100, 100, 8]},
#          8: {"role": 5, "attr": [1, 1, 1, 1, 1, 7]}, 9: {"role": 7, "attr": [100, 100, 100, 100, 100, 7]},
#          10: {"role": 5, "attr": [1, 1, 1, 1, 1, 7]}, 11: {"role": 3, "attr": [100, 100, 100, 100, 100, 7]},
#          12: {"role": 5, "attr": [1, 1, 1, 1, 1, 7]}, 13: {"role": 3, "attr": [100, 100, 100, 100, 100, 7]},
#          14: {"role": 5, "attr": [1, 1, 1, 1, 1, 7]}, 15: {"role": 3, "attr": [100, 100, 100, 100, 100, 7]},
#          16: {"role": 5, "attr": [1, 1, 1, 1, 1, 7]}, 17: {"role": 3, "attr": [100, 100, 100, 100, 100, 7]}
#          }


attrs = {0: {"role": 0, "attr": [1, 4]}, 1: {"role": 1, "attr": [1.2, 4]},
         2: {"role": 0, "attr": [1, 4]}, 3: {"role": 1, "attr": [1.2, 4]},
         4: {"role": 2, "attr": [1, 3]}, 5: {"role": 2, "attr": [1, 3]},
         6: {"role": 4, "attr": [1.2, 3]}, 7: {"role": 5, "attr": [1, 3]},
         8: {"role": 3, "attr": [1.2, 3]}, 9: {"role": 3, "attr": [1.2, 3]}
         }

for i in range(1, 11):
    node_id = 9 + i
    if node_id % 2 == 0:
        attrs[node_id] = {"role": 2, "attr": [1, 3]}
    else:
        attrs[node_id] = {"role": 3, "attr": [1.2, 3]}

nx.set_node_attributes(G, attrs)
role_id = []
for i in range(20):
    role_id.append(G.nodes[i]['role'])

plot_networkx(G, role_id)

nx.write_gpickle(G, "intro.gpickle")
np.savetxt('intro.out', np.array(role_id))