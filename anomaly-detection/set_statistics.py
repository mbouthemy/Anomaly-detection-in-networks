# Set of the statistics used for the NetEDM module and compute netEMD.
from scipy.optimize import fmin
import scipy.integrate as integrate
import math
import networkx as nx
import numpy as np


def netEMD(statistic, graph_1, graph_2):
    """Compute the netEMD between two graphs given a precise statistic."""
    r = fmin(lambda s: function_to_optimize(s, statistic, graph_1, graph_2), 1.0)
    print(r)
    return r


def function_to_optimize(s, statistic, graph_1, graph_2):
    """Define the function to get the minimum."""
    # We compute the integral from -inf to inf.
    integral = integrate.quad(lambda x: abs(empirical_distribution_function(statistic, x + s, graph_1) -
                                            empirical_distribution_function(statistic, x, graph_2)), -np.inf,
                              np.inf)
    return integral[0]


def empirical_distribution_function(statistic, x, graph):
    """
    Compute the empirical distribution function.
    """
    total_sum = 0
    emp_variance = variance_statistic(statistic, graph)
    for j in range(nx.number_of_nodes(graph)):
        if x <= statistic(j, graph) / emp_variance:
            total_sum += 1
    return total_sum / nx.number_of_nodes(graph)


def mean_statistic(statistic, graph):
    """Get the mean of a given statistic."""
    total_sum = 0
    for node in nx.nodes(graph):
        total_sum += statistic(node, graph)
    return float(total_sum / nx.number_of_nodes(graph))


def variance_statistic(statistic, graph):
    """Compute the variance of the statistic."""
    total_sum, n = 0, nx.number_of_nodes(graph)
    mean = mean_statistic(statistic, graph)
    for i in range(n):
        total_sum += (statistic(i, graph) - mean)**2
    return total_sum / n


def statistics_1(node, graph):
    """In-strength statistics"""
    total_weight = 0
    for x in graph.predecessors(node):
        total_weight += graph[x][node]['weight']
    return total_weight


def statistics_2(node, graph):
    """Out-strength statistic."""
    total_weight = 0
    for x in graph.successors(node):
        total_weight += graph[x][node]['weight']
    return total_weight


def statistics_3(node, graph):
    """Sum of the in-strength and out-strength."""
    return statistics_1(node, graph) + statistics_2(node, graph)


def statistics_4(node, graph):
    """Statistic with three motif."""
    return 0

# TODO: Set the other statistics.
