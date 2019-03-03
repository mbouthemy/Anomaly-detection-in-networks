# Path finder implementation
import heapq
from itertools import count


def paths(graph, beam_width):
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
