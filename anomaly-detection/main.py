import numpy as np
import random
import math
from utils import compute_product
from graph import Graph


def main():
    n = 1000
    p = 0.5
    weight = 0.7
    graph = Graph(n, p, weight)
    graph.create_graph()
    graph.selection_of_anomalies()
    graph.info_anomalies()
    graph.test()
    list_of_anomalies = selection_of_anomalies()
    #information_anomalies(list_of_anomalies)
    #print(np.matrix(graph))
    #print("\n blabla \n")
    list_of_nodes = insert_ring(graph, 7, n)
    list_of_clique = insert_clique(graph, 7, n)
    list_of_star = insert_star(graph, 7, n)
    list_of_tree = insert_tree(graph, 9, n)
    a, b, c = gaw_statistics(graph, 1, n)
    print(a,b,c)
    #print(np.matrix(graph))
    print("ta mere")
    return 0


def creation_er_graph(n, p):
    """ Create the ER graph."""
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_bernoulli = np.random.uniform()
            if x_bernoulli <= p:
                graph[i, j] = np.random.uniform()
            else:
                graph[i, j] = 0

            if i == j:  # same node
                graph[i, j] = 0
    return graph


def selection_of_anomalies():
    """ Get the different anomalies and their size."""
    number_of_anomalies = np.random.randint(5, 21)  # 21 is excluded
    list_of_anomalies = []

    for k in range(number_of_anomalies):
        type_of_anomality = np.random.randint(0, 5)
        size = np.random.randint(5, 21)  # draw size of anomaly
        size_tree = 9
        name_of_anomalies = ["ring", "path", "clique", "star", "tree"]

        if type_of_anomality == 4:
            list_of_anomalies.append([name_of_anomalies[4], size_tree])
        else:
            list_of_anomalies.append([name_of_anomalies[type_of_anomality], size])

    return list_of_anomalies


def information_anomalies(list_of_anomalies):
    for x in list_of_anomalies:
        print(x)


def insert_ring(graph, size_of_ring,  n):
    """Insertion of a ring into the graph."""
    weight = 0.7
    ring_nodes = random.sample(range(n), size_of_ring)  # Select the nodes of the ring
    for j in range(size_of_ring - 1):
        graph[ring_nodes[j], ring_nodes[j+1]] = np.random.uniform(weight)

    # Connect the last one to the first one
    graph[ring_nodes[-1], ring_nodes[0]] = np.random.uniform(weight)

    return ring_nodes


def insert_direct_path(graph, size_of_path, n):
    """ Insertion of a direct path into the graph."""
    weight = 0.7
    path_nodes = random.sample(range(n), size_of_path)  # Select the nodes
    for j in range(size_of_path - 1):
        graph[path_nodes[j], path_nodes[j + 1]] = np.random.uniform(weight)

    return path_nodes


def insert_clique(graph, size_of_clique, n):
    """Insertion of a clique in the graph."""
    weight = 0.7
    clique_nodes = random.sample(range(n), size_of_clique)  # Select the nodes

    # We do a loop and alternatively we place in and out vertices
    for i in range(size_of_clique - 1):
        for j in range(i+1, size_of_clique):
            if (j-i-1) % 2 == 0:  # outgoing from i to j
                graph[clique_nodes[i], clique_nodes[j]] = np.random.uniform(weight)
            else:  # from j to i
                graph[clique_nodes[j], clique_nodes[i]] = np.random.uniform(weight)

    return clique_nodes


def insert_star(graph, size_of_star, n):
    """Insertion of a star scheme into the graph."""
    weight = 0.7
    star_nodes = random.sample(range(n), size_of_star)  # Select the nodes
    middle = int(size_of_star / 2)

    # Ingoing vertices to the middle
    for k in range(middle):
        graph[star_nodes[k], star_nodes[middle]] = np.random.uniform(weight)

    # Outgoing from the middle
    for k in range(middle+1, size_of_star):
        graph[star_nodes[middle], star_nodes[k]] = np.random.uniform(weight)

    return star_nodes


def insert_tree(graph, size_tree, n):
    """Insert tree in the graph, we suppose that size tree is 9."""
    weight = 0.7
    tree_nodes = random.sample(range(n), size_tree)  # Select the nodes

    for i in range(5):  # First shell
        for k in range(3):
            graph[tree_nodes[i], tree_nodes[5+k]] = np.random.uniform(weight)

    for i in range(3):  # Second shell connection
        graph[tree_nodes[5+i], tree_nodes[8]] = np.random.uniform(weight)

    return tree_nodes


def gaw_statistics(graph, node, n):
    """Compute the three gaw statistics for a single node."""
    degree = 0
    list_of_weights = []
    for i in range(n):
        if graph[i, node] != 0:  # Ingoing node
            degree += 1
            list_of_weights.append(graph[i, node])
        if graph[node, i] != 0:  # Outgoing node
            degree += 1
            list_of_weights.append(graph[node, i])

    list_of_weights.sort()   # Order the list

    gaw_score = (compute_product(list_of_weights, 0))**(1/degree)
    gaw_score_10 = (compute_product(list_of_weights, degree - math.ceil(0.1*degree) + 1))**(1/math.ceil(0.1*degree))
    gaw_score_20 = (compute_product(list_of_weights, degree - math.ceil(0.2*degree) + 1))**(1/math.ceil(0.2*degree))

    return gaw_score, gaw_score_10, gaw_score_20


if __name__ == "__main__":
    main()
