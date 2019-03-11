####################################################################
#
#                   NET_EMD FEATURES (3.5)
#
#  Functions to compute the NetEMD features for a networkx as in the paper
#  Use the function create_features_net_emd to get all the features for a network
#
####################################################################



import networkx as nx
import numpy as np
from generation import generate_null
from utils import trimmed_mean
from bisect import bisect
import pandas as pd
from scipy import special
from set_statistics import statistics_1 as stat_1
from set_statistics import statistics_2 as stat_2


def create_features_net_emd(graph, number_monte_carlo=200):
    """Create the features based on the net EMD detection. (3.4)"""

    # Change the name here to add more statistics.
    set_of_statistics = [stat_1, stat_2]
    G_stat, G_stat_non_normalized = compute_statistic(graph, set_of_statistics)

    # Create the reference set and the null set
    reference_set = node_level_creation_set(graph, set_of_statistics, size=15)
    null_set = node_level_creation_set(graph, set_of_statistics, size=number_monte_carlo)

    # Compute the net EMD of the real graph.
    y_ref = get_trimmed_mean_net_emd(G_stat, reference_set)

    # Compute the net EMD of the null sets and get the p-values accordingly.
    p_values = find_p_value_of_each_statistics(y_ref, reference_set, null_set)

    # Calculate the data frame based on the two scores formulas.
    df = calculate_score_nodes(graph, set_of_statistics, p_values, G_stat_non_normalized)
    print("\nThe features for the Net EMD (3.4) have been created.")
    return df


def compute_statistic(graph, set_of_statistics):
    """Compute the statistics for each node in the graph.
    Return a matrix with node as lines and statistics as columns."""
    matrix_statistic = np.zeros((nx.number_of_nodes(graph), len(set_of_statistics)))
    matrix_statistic_normalized = np.zeros((nx.number_of_nodes(graph), len(set_of_statistics)))

    for j in range(len(set_of_statistics)):
        stat = set_of_statistics[j]  # Get the statistic.
        for k in range(len(nx.nodes(graph))):
            matrix_statistic[k, j] = stat(list(graph.nodes)[k], graph)  # Compute the statistic for the node.
        var = np.var(matrix_statistic[:, j])
        for k in range(len(nx.nodes(graph))):
            matrix_statistic_normalized[k, j] = matrix_statistic[k, j] / var
    return matrix_statistic_normalized, matrix_statistic


def net_emd(matrix_stat_1, matrix_stat_2):
    """Compute the netEMD between two graphs, given the matrix of statistics."""

    y = []  # Initialize the vector y.
    # Get the list of statistics in each node for both graph.

    number_statistic = len(matrix_stat_1[0])
    number_nodes = len(matrix_stat_1)
    for i in range(number_statistic):  # For each statistic
        list_1 = matrix_stat_1[:, i].reshape((number_nodes, 1))  # Get the list of statistics and reshape it.
        # We add a tag to facilitate the computation of the later integral.
        list_1_tag = np.concatenate((list_1, np.zeros((number_nodes, 1))), axis=1)

        list_2 = matrix_stat_2[:, i].reshape((number_nodes, 1))
        list_2_tag = np.concatenate((list_2, np.ones((number_nodes, 1))), axis=1)

        # Get the best value for s.
        min_value = float('inf')  # Initialization.

        # Test on multiple values of s to try to get the minimum.
        for s in np.arange(-5.0, 5., 0.1):
            value = compute_integral(list_1_tag, list_2_tag, s)
            if min_value > value:
                min_value = value
        y.append(min_value)

    return y


def compute_integral(list_1, list_2, s):
    """Compute the integral with the parameter s."""
    # We set a tag to distinguish the two functions.
    z = np.array(list_1)
    z[:, 0] = z[:, 0] - s
    z = z.tolist()  # Use s to shift the index
    t = z + list_2.tolist()    # Concatenate both list
    t.sort()  # Order the list
    U_1, U_2 = 1, 1  # Initialize the value of sum of indicators.
    sum_integral = 0

    for j in range(len(t) - 1):
        if t[j][1] == 0:  # If the tag corresponds to the first graph.
            U_1 -= 1 / (len(t) * 0.5)  # Decrease the indicator function.
        else:
            U_2 -= 1 / (len(t) * 0.5)
        # Here is the formula for computation of the local part of the integral.
        sum_integral += abs(U_1 - U_2) * (t[j + 1][0] - t[j][0])

    return sum_integral


def node_level_creation_set(graph, set_of_statistics, size):
    """Compute the node level statistics for the 15 reference sets."""
    set_of_null = []
    for i in range(size):
        reference_graph = generate_null(graph)
        node_level_stat, node_level_stat_non_normalized = compute_statistic(reference_graph, set_of_statistics)
        set_of_null.append(node_level_stat)
    print("The node level statistics has been computed for {} graphs.".format(size))
    return set_of_null


def get_trimmed_mean_net_emd(graph_statistics, reference_set_statistics):
    """Compute the trimmed mean of the net EMD between a graph and a reference set."""

    number_reference = len(reference_set_statistics)
    number_statistic = len(graph_statistics[0])
    y_mean = np.ones((number_reference, number_statistic))  # Initialize with the shape of the matrix.
    y = np.empty(number_statistic)

    for j in range(number_reference):  # For each reference.
        y_mean[j, :] = net_emd(graph_statistics, reference_set_statistics[j])  # We compute the statistic.

    for k in range(number_statistic):  # Now we apply the trimmed mean for each of the statistic.
        y[k] = trimmed_mean(y_mean[:, k])
    return y


def find_p_value_of_each_statistics(y_ref, reference_set, null_set):
    """Compute the NetEMD between the reference set and the null set.
    We also find the rank of the statistics and then the p-value for each."""
    number_null = len(null_set)
    number_statistic = len(null_set[0][0])
    y_null = np.zeros((number_null, number_statistic))  # Create the matrix to keep y of the null simulations.
    for i in range(number_null):
        print("Compute net EMD {} / {} of the null set.   ".format(i+1, number_null), end="\r")
        null_stat = null_set[i]
        y_null[i, :] = get_trimmed_mean_net_emd(null_stat, reference_set)

    rank_of_statistic = np.empty(len(y_ref))  # Create the array which contains the rank of each statistic.

    for i in range(len(y_ref)):
        values_stat = y_null[:, i]
        values_stat.sort()  # Sort it.
        rank_of_statistic[i] = bisect(values_stat, y_ref[i])  # Get the rank of the statistic.
    print("\nThe p-value of each statistics has been computed.")
    p_value = 1 / rank_of_statistic
    return p_value


def score_1(list_of_node_stat):
    """Compute the score 1 given the list of all the nodes"""
    var, mean = np.var(list_of_node_stat), np.mean(list_of_node_stat)
    n = len(list_of_node_stat)
    list_of_scores = np.empty(n)

    for j in range(n):
        threshold = abs((list_of_scores[j] - mean) / var)
        if threshold < 2:
            list_of_scores[j] = 0
        else:
            list_of_scores[j] = threshold

    return list_of_scores


def score_2(list_of_node_stat, p_value):
    """Compute the score 2 based on the list of nodes statistics and the p-value."""
    var, mean = np.var(list_of_node_stat), np.mean(list_of_node_stat)
    n = len(list_of_node_stat)
    list_of_scores = np.empty(n)
    standardized_list = abs((list_of_node_stat - mean) / var)
    threshold = np.percentile(standardized_list, 95)

    for j in range(n):
        if standardized_list[j] < threshold:
            list_of_scores[j] = 0
        else:
            list_of_scores[j] = special.ndtri(1 - p_value)  # Inverted CDF of gaussian.

    return list_of_scores


def calculate_score_nodes(G, set_of_statistics, p_values, node_statistics_non_normalized):
    """Assign a score for each nodes based on."""
    df = pd.DataFrame(index = G.nodes())
    number_nodes = len(node_statistics_non_normalized)
    for j in range(len(set_of_statistics)):  # For each statistics
        name = set_of_statistics[j].__name__

        if p_values[j] > 0.05:
            df[str(name) + '_score_1'] = number_nodes * [0]
            df[str(name) + '_score_2'] = number_nodes * [0]
        else:
            df[str(name) + '_score_1'] = score_1(node_statistics_non_normalized[:, j])
            df[str(name) + '_score_2'] = score_2(node_statistics_non_normalized[:, j], p_values[j])
    return df

