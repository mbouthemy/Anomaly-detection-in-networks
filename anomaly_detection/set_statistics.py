# Set of the statistics used for the NetEDM module and compute netEMD.


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
        total_weight += graph[node][x]['weight']
    return total_weight


def statistics_3(node, graph):
    """Sum of the in-strength and out-strength."""
    return statistics_1(node, graph) + statistics_2(node, graph)


def statistics_4(node, graph):
    """Statistic with three motif."""
    return 0

# TODO: Set the other statistics.
