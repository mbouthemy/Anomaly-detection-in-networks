# utils.py
import numpy as np


def add_weight(direct_graph):
    """Add weight to the existing Erdos Renyi Graph."""
    for e in direct_graph.edges():
        direct_graph[e[0]][e[1]]['weight'] = np.random.uniform()


def compute_product(list_of_elements, begin):
    """Compute the product of a list"""
    product = 1
    for j in range(begin, len(list_of_elements)):
        product *= list_of_elements[j]
    return product
