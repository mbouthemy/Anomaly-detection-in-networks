import numpy as np
import random
import math
from utils import compute_product


class Graph:

    def __init__(self, number_nodes, probability, weight):
        self.number_nodes = number_nodes
        self.probability = probability
        self.weight = weight
        self.matrix = None
        self.anomalies = None

    def create_graph(self):
        """ Create the ER graph."""
        graph = np.zeros((self.number_nodes, self.number_nodes))
        for i in range(self.number_nodes):
            for j in range(self.number_nodes):
                x_bernoulli = np.random.uniform()
                if x_bernoulli <= self.probability:
                    graph[i, j] = np.random.uniform()
                else:
                    graph[i, j] = 0

                if i == j:  # same node
                    graph[i, j] = 0
        self.matrix = graph

    def selection_of_anomalies(self):
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

        self.anomalies = list_of_anomalies

    def info_anomalies(self):
        for x in self.anomalies:
            print(x)

    def insert_ring(self, size_of_ring):
        """Insertion of a ring into the graph."""
        print("Dans ring")
        ring_nodes = random.sample(range(self.number_nodes), size_of_ring)  # Select the nodes of the ring
        for j in range(size_of_ring - 1):
            self.matrix[ring_nodes[j], ring_nodes[j + 1]] = np.random.uniform(self.weight)

        # Connect the last one to the first one
        self.matrix[ring_nodes[-1], ring_nodes[0]] = np.random.uniform(self.weight)

    def insert_direct_path(self, size_of_path):
        """ Insertion of a direct path into the graph."""
        path_nodes = random.sample(range(self.number_nodes), size_of_path)  # Select the nodes
        for j in range(size_of_path - 1):
            self.matrix[path_nodes[j], path_nodes[j + 1]] = np.random.uniform(self.weight)

    def insert_clique(self, size_of_clique):
        """Insertion of a clique in the graph."""
        clique_nodes = random.sample(range(self.number_nodes), size_of_clique)  # Select the nodes

        # We do a loop and alternatively we place in and out vertices
        for i in range(size_of_clique - 1):
            for j in range(i + 1, size_of_clique):
                if (j - i - 1) % 2 == 0:  # outgoing from i to j
                    self.matrix[clique_nodes[i], clique_nodes[j]] = np.random.uniform(self.weight)
                else:  # from j to i
                    self.matrix[clique_nodes[j], clique_nodes[i]] = np.random.uniform(self.weight)

    def insert_star(self, size_of_star):
        """Insertion of a star scheme into the graph."""
        star_nodes = random.sample(range(self.number_nodes), size_of_star)  # Select the nodes
        middle = int(size_of_star / 2)

        # Ingoing vertices to the middle
        for k in range(middle):
            self.matrix[star_nodes[k], star_nodes[middle]] = np.random.uniform(self.weight)

        # Outgoing from the middle
        for k in range(middle + 1, size_of_star):
            self.matrix[star_nodes[middle], star_nodes[k]] = np.random.uniform(self.weight)

    def insert_tree(self, size_tree):
        """Insert tree in the graph, we suppose that size tree is 9."""
        tree_nodes = random.sample(range(self.number_nodes), size_tree)  # Select the nodes

        for i in range(5):  # First shell
            for k in range(3):
                self.matrix[tree_nodes[i], tree_nodes[5 + k]] = np.random.uniform(self.weight)

        for i in range(3):  # Second shell connection
            self.matrix[tree_nodes[5 + i], tree_nodes[8]] = np.random.uniform(self.weight)

        return tree_nodes

    def insert_anomalies(self):
        for x in self.anomalies:
            name_anomaly, size_anomaly = x[0], x[1]
            if name_anomaly == 'ring':
                self.insert_ring(size_anomaly)
            elif name_anomaly == 'path':
                self.insert_direct_path(size_anomaly)
            elif name_anomaly == 'clique':
                self.insert_clique(size_anomaly)
            elif name_anomaly == 'star':
                self.insert_star(size_anomaly)
            else:
                self.insert_tree(size_anomaly)
        print('Anomalies inserted')

    def show_graph(self):
        print(np.matrix(self.matrix))

