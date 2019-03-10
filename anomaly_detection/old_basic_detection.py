# Basic detection (3.1)

import numpy as np
from scipy import special
import math
from utils import compute_product


def get_score_nodes(matrix, number_monte_carlo):
    """Get scores for each nodes."""
    nodes_stat = monte_carlo_gaw(matrix, number_monte_carlo)  # Get the statistics for each node.
    number_nodes = len(matrix)
    score_nodes = np.zeros(number_nodes)
    for node in range(number_nodes):
        stat = nodes_stat[node]
        mean_stat, sd_stat = np.mean(stat), np.std(stat)
        z_score = (gaw_statistics(matrix, node) - mean_stat) / sd_stat
        p_value = 1 - special.ndtr(z_score)  # Compute the p-value
        if p_value > 0.05:
            score_nodes[node] = 0
        else:
            score_nodes[node] = special.ndtri(1 - p_value)  # Inverted CDF of gaussian.

    return score_nodes


def monte_carlo_gaw(matrix, number_monte_carlo):
    """Run the Monte Carlo algorithm M times to get the gaw statistics for each node. """
    number_nodes = len(matrix)
    non_zero_position = get_non_zero_position(matrix)
    nodes_stat = np.empty([number_nodes, number_monte_carlo])  # Create the list for statistics for each node.

    for j in range(number_monte_carlo):
        matrix_permuted = get_permuted_graph(matrix, non_zero_list=non_zero_position)
        for node in range(number_nodes):
            nodes_stat[node, j] = gaw_statistics(matrix_permuted, node)

    return nodes_stat


def gaw_statistics(matrix, node):
    """Compute the three gaw statistics for a single node."""
    degree = 0
    n = len(matrix[0])
    list_of_weights = []
    for i in range(n):
        if matrix[i, node] != float(0):  # Ingoing node
            degree += 1
            list_of_weights.append(matrix[i, node])
        if matrix[node, i] != float(0):  # Outgoing node
            degree += 1
            list_of_weights.append(matrix[node, i])

    list_of_weights.sort()   # Order the list

    gaw_score = (compute_product(list_of_weights, 0))**(1/degree)
    # gaw_score_10 = (compute_product(list_of_weights, degree - math.ceil(0.1*degree) + 1))**(1/math.ceil(0.1*degree))
    # gaw_score_20 = (compute_product(list_of_weights, degree - math.ceil(0.2*degree) + 1))**(1/math.ceil(0.2*degree))

    return gaw_score


def get_non_zero_position(matrix):
    """Get the positions of fixed nodes (i.e non zeros)."""
    non_zero_list = []
    array = matrix.flatten()  # Transform the matrix to a list
    for i in range(len(array)):
        if array[i] != float(0):
            non_zero_list.append(i)
    return non_zero_list


def get_permuted_graph(matrix, non_zero_list):
    """Permute the vertices of the graph but keep nodes in place."""
    array = matrix.flatten()
    new_array = array.copy()
    permuted_index = non_zero_list.copy()
    np.random.shuffle(permuted_index)
    for i in range(len(non_zero_list)):
        new_array[non_zero_list[i]] = array[permuted_index[i]]
    n = int(math.sqrt(len(new_array)))
    new_matrix_permuted = np.reshape(new_array, (n, n))
    return new_matrix_permuted
