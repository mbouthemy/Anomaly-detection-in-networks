import numpy as np


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

    def test(self):
        print("blabla")
