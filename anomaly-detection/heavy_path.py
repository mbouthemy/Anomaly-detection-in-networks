# Perform the augmentation and the community detection with the Python Louvain algorithm.
import numpy as np


def weight_distribution_percentile(direct_graph):
    """Get the 99th-percentile of the weight distribution over the graph."""
    list_of_weight = []
    for (u, v, d) in direct_graph.edges(data=True):
        list_of_weight.append(d['weight'])
    threshold = np.percentile(list_of_weight, 99)  # Get the 99-th percentile
    return threshold


def augmentation(direct_graph):
    """Find heavy path in the graph and augment those path."""
    threshold = weight_distribution_percentile(direct_graph)
    for (u, v, d) in direct_graph.edges(data=True):
        if d['weight'] >= threshold:  # First Edge
            w_1 = d['weight']
            successors = direct_graph.successors(v)  # Get all successors of v
            for w in successors:
                if direct_graph[v][w]['weight'] >= threshold and u != w:  # Second Edge
                    w_2 = direct_graph[v][w]['weight']

                    if direct_graph.has_edge(u, w):
                        direct_graph[u][w]['weight'] = max(direct_graph[u][w]['weight'], min(w_1, w_2))
                        # Update is min(w_3, max(w_1, w_2))

                    else:
                        direct_graph.add_weighted_edges_from([(u, w, min(w_1, w_2))])
