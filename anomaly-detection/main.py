import numpy as np
import random
from scipy import special
import math
import utils
from old_graph import Graph
import community
import networkx as nx
import matplotlib.pyplot as plt
import anomalies
import heavy_path


def main():
    w = 0.7
    p = 0.05
    number_nodes = 50

    DG = nx.erdos_renyi_graph(number_nodes, p, seed=2, directed=True)
    utils.add_weight(DG)

    nx.draw(DG)
    plt.show()

    list_of_anomalies = anomalies.selection_of_anomalies()
    anomalies.info_anomalies(list_of_anomalies)
    anomalies.insert_anomalies(DG, list_of_anomalies, w)

    nx.draw(DG)
    plt.show()

    heavy_path.augmentation(DG)

    return 0


if __name__ == "__main__":
    main()
