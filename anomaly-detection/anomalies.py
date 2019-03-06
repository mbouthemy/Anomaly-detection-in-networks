# For insertion of anomalies based on the Graph from networkX

import numpy as np
import random
import networkx as nx
import pandas as pd


def selection_of_anomalies():
    """ Get the different anomalies and their size."""
    number_of_anomalies = np.random.randint(5, 21)  # 21 is excluded
    list_of_anomalies = []
    for k in range(number_of_anomalies):
        type_of_anomaly = np.random.randint(0, 5)
        size = np.random.randint(5, 21)  # draw size of anomaly
        size_tree = 9
        name_of_anomalies = ["ring", "path", "clique", "star", "tree"]

        if type_of_anomaly == 4:  # size of tree is constant
            list_of_anomalies.append([name_of_anomalies[4], size_tree])
        else:
            list_of_anomalies.append([name_of_anomalies[type_of_anomaly], size])

    return list_of_anomalies


def info_anomalies(list_of_anomalies):
    for x in list_of_anomalies:
        print(x)


def insert_anomalies(direct_graph, list_of_anomalies, w):
    df_anomaly = pd.DataFrame(np.zeros(nx.number_of_nodes(direct_graph)), columns=['anomaly'])
    for x in list_of_anomalies:
        name_anomaly, size_anomaly = x[0], x[1]
        if name_anomaly == 'ring':
            insert_ring(direct_graph, size_anomaly, w, df_anomaly)
        elif name_anomaly == 'path':
            insert_direct_path(direct_graph, size_anomaly, w, df_anomaly)
        elif name_anomaly == 'clique':
            insert_clique(direct_graph, size_anomaly, w, df_anomaly)
        elif name_anomaly == 'star':
            insert_star(direct_graph, size_anomaly, w, df_anomaly)
        else:
            insert_tree(direct_graph, size_anomaly, w, df_anomaly)
    print('Anomalies inserted')
    return df_anomaly


def insert_ring(direct_graph, size_of_ring, w, df):
    """Insertion of a ring into the graph."""
    ring_nodes = random.sample(range(nx.number_of_nodes(direct_graph)), size_of_ring)  # Select the nodes of the ring
    for j in range(size_of_ring - 1):
        weight = np.random.uniform(w)
        direct_graph.add_weighted_edges_from([(ring_nodes[j], ring_nodes[j+1], weight)])

    # Connect the last one to the first one
    weight = np.random.uniform(w)
    direct_graph.add_weighted_edges_from([(ring_nodes[-1], ring_nodes[0], weight)])
    df.iloc[ring_nodes] = 1  # Specify the nodes who are concerned with the anomaly.


def insert_direct_path(direct_graph, size_of_path, w, df):
    """ Insertion of a direct path into the graph."""
    path_nodes = random.sample(range(nx.number_of_nodes(direct_graph)), size_of_path)  # Select the nodes
    for j in range(size_of_path - 1):
        weight = np.random.uniform(w)
        direct_graph.add_weighted_edges_from([(path_nodes[j], path_nodes[j+1], weight)])

    df.iloc[path_nodes] = 1  # Specify the nodes who are concerned with the anomaly.


def insert_clique(direct_graph, size_of_clique, w, df):
    """Insertion of a clique in the graph."""
    clique_nodes = random.sample(range(nx.number_of_nodes(direct_graph)), size_of_clique)  # Select the nodes

    # We do a loop and alternatively we place in and out vertices
    for i in range(size_of_clique - 1):
        for j in range(i + 1, size_of_clique):
            weight = np.random.uniform(w)
            if (j - i - 1) % 2 == 0:  # outgoing from i to j
                direct_graph.add_weighted_edges_from([(clique_nodes[i], clique_nodes[j], weight)])
            else:  # from j to i
                direct_graph.add_weighted_edges_from([(clique_nodes[j], clique_nodes[i], weight)])
    df.iloc[clique_nodes] = 1  # Specify the nodes who are concerned with the anomaly.


def insert_star(direct_graph, size_of_star, w, df):
    """Insertion of a star scheme into the graph."""
    star_nodes = random.sample(range(nx.number_of_nodes(direct_graph)), size_of_star)  # Select the nodes
    middle = int(size_of_star / 2)

    # Ingoing vertices to the middle
    for k in range(middle):
        weight = np.random.uniform(w)
        direct_graph.add_weighted_edges_from([(star_nodes[k], star_nodes[middle], weight)])

    # Outgoing from the middle
    for k in range(middle + 1, size_of_star):
        weight = np.random.uniform(w)
        direct_graph.add_weighted_edges_from([(star_nodes[middle], star_nodes[k], weight)])

    df.iloc[star_nodes] = 1  # Specify the nodes who are concerned with the anomaly.


def insert_tree(direct_graph, size_tree, w, df):
    """Insert tree in the graph, we suppose that size tree is 9."""
    tree_nodes = random.sample(range(nx.number_of_nodes(direct_graph)), size_tree)  # Select the nodes

    for i in range(5):  # First shell
        for k in range(3):
            weight = np.random.uniform(w)
            direct_graph.add_weighted_edges_from([(tree_nodes[i], tree_nodes[5 + k], weight)])

    for i in range(3):  # Second shell connection
        weight = np.random.uniform(w)
        direct_graph.add_weighted_edges_from([(tree_nodes[5 + i], tree_nodes[8], weight)])
    df.iloc[tree_nodes] = 1  # Specify the nodes who are concerned with the anomaly.
