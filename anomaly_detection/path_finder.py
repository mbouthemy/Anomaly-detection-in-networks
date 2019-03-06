# Path finder implementation
import heapq
import numpy as np
import generation
import utils
from scipy import special
from itertools import count
import pandas as  pd


def create_features_path_finder(graph, beam_width, number_monte_carlo, number_to_keep):
    """Create features for the path finder (3.5)
    beam_width = 100
    number_monte_carlo = 100  # Specify the number of Monte Carlo simulation.
    number_to_keep = 20  # Specify the number of paths to keep at each iteration of size.
    """
    # Get the Monte Carlo statistics (that's a matrix).
    monte_carlo_stat_path = create_monte_carlo_statistics(graph, number_monte_carlo, beam_width, number_to_keep)

    df = pd.DataFrame()  # Create the empty data frame
    paths_to_consider = create_paths(graph, beam_width)

    for j in range(19):
        paths_to_consider = keep_top_path(paths_to_consider, number_to_keep)
        feature_nodes = calculate_node_statistics(graph, paths_to_consider, monte_carlo_stat_path, j)
        df['path_size_' + str(j + 3)] = feature_nodes
        paths_to_consider = increase_path_size(paths_to_consider, graph)
    print("Features for path finder (3.5) inserted.")
    return df


def calculate_node_statistics(graph, path_to_consider, monte_carlo_stat, current_feature):
    """Compute the p-value for the path finder statistics, and return features"""

    features = np.array([0.] * graph.number_of_nodes())

    for x in path_to_consider:
        weight, path = x[0], x[2]

        # We compute the p-value with the statistics from Monte Carlo simulation.
        p_value = utils.p_val_lower(weight, monte_carlo_stat[:, current_feature])
        # If the p-value is important, we assert a value for every node in that path.
        if p_value > 0.05:
            features[path] = special.ndtri(1 - p_value)  # Inverted CDF of gaussian

    return features


def create_monte_carlo_statistics(graph, number_monte_carlo, beam_width, number_to_keep):
    """We store the statistics in a matrix. Each row corresponds to another simulation, and columns to the
    20 statistics for each path."""
    number_path = 19
    monte_carlo_statistics = np.zeros((number_monte_carlo, number_path))

    for T in range(number_monte_carlo):

        graph_null = generation.generate_null(graph)  # Generate the null graph
        paths = create_paths(graph_null, beam_width)  # Initialise the path
        paths = keep_top_path(paths, number_to_keep)

        largest_weight = max(paths)[0]  # Get the largest weight.
        monte_carlo_statistics[T, 0] = largest_weight

        for i in range(1, number_path):  # Because we start with path of size 3
            paths = increase_path_size(paths, graph)
            paths = keep_top_path(paths, number_to_keep)
            largest_weight = max(paths)[0]  # Get the largest weight.
            monte_carlo_statistics[T, i] = largest_weight

    return monte_carlo_statistics


def keep_top_path(paths, number_to_keep):
    """
    Keep only the top beam width paths in the heap.
    """
    for i in range(len(paths) - number_to_keep):
        heapq.heappop(paths)

    return paths


def increase_path_size(paths, graph):
    """
    Increase the size of the paths of 1.
    """
    new_list_of_paths = []
    cnt = count()  # Counter is useful for the heap insertion.

    for x in paths:
        path = x[2]  # Get the path
        first_node, last_node = path[0], path[-1]

        for u, v, d in graph.in_edges(first_node, data=True):
            if u not in path:  # We don't want a loop
                new_path = [u] + path
                new_weight = x[0] + d['weight']  # Add the weight to the new
                heapq.heappush(new_list_of_paths, (new_weight, next(cnt), new_path))

        for u, v, d in graph.out_edges(last_node, data=True):
            if v not in path:
                new_path = path + [v]
                new_weight = x[0] + d['weight']  # Add the weight to the new
                heapq.heappush(new_list_of_paths, (new_weight, next(cnt), new_path))

    return new_list_of_paths


def create_paths(graph, beam_width):
    """
    Create the list of paths of size three to consider.

    :param graph: The graph based on NetworkX.
    :param beam_width: The number of paths we want.
    :return: The list of paths to consider.
    """
    cnt = count()  # Counter is useful for the heap insertion.

    paths_to_consider = []  # This is a heap

    for j in range(beam_width):
        heapq.heappush(paths_to_consider, (0, next(cnt), [1, 2, 3]))  # Initialise with dummy paths

    # TODO: Check the case where there is no edge.
    for current_node in list(graph.nodes):
        max_weight_in = 0.
        max_weight_out = 0.
        current_out = None
        current_in = None

        for u, v, d in graph.in_edges(current_node, data=True):
            if d['weight'] > max_weight_in:
                max_weight_in = d['weight']
                current_in = u

        for u, v, d in graph.out_edges(current_node, data=True):
            if d['weight'] > max_weight_out:
                max_weight_out = d['weight']
                current_out = v
        fitness_of_path = max_weight_out + max_weight_in

        # If the fitness is inferior to the smallest weight
        if fitness_of_path > heapq.nsmallest(1, paths_to_consider)[0][0]:
            # We add it to the heap and pop the smallest one
            heapq.heappushpop(paths_to_consider,
                              (fitness_of_path, next(cnt), [current_in, current_node, current_out]))

    return paths_to_consider
