####################################################################
#
#                   AUGMENTATION OF NETWORKS
#
#  Perform the augmentation of a network as described in the paper.
#
####################################################################


# Imports
import numpy as np


def weight_distribution_percentile(direct_graph, percentile = 99):
    """Get the 99th-percentile of the weight distribution over the graph."""
    s_in, s_out, W = zip(*direct_graph.edges(data = "weight"))
    threshold = np.percentile(W, percentile)  # Get the 99-th percentile
    return threshold


def augmentation(direct_graph):
    """Find heavy path in the graph and augment those path."""
    heavy_graph = direct_graph.copy()
    threshold = weight_distribution_percentile(direct_graph)
    for (u, v, w) in direct_graph.edges(data="weight"):
        if w >= threshold:  # First Edge
            w_1 = w
            successors = direct_graph.successors(v)  # Get all successors of v
            for w in successors:
                if direct_graph[v][w]['weight'] >= threshold and u != w:  # Second Edge
                    w_2 = direct_graph[v][w]['weight']

                    if direct_graph.has_edge(u, w):
                        w_3 = direct_graph[u][w]['weight']
                        heavy_graph[u][w]['weight'] = max(w_3, min(w_1, w_2))
                        # Update is min(w_3, max(w_1, w_2))

                    else:
                        heavy_graph.add_weighted_edges_from([(u, w, min(w_1, w_2))])
    return heavy_graph
