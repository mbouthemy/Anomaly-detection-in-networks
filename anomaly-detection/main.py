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
import path_finder


def main():
    w = 0.7
    p = 0.05
    number_nodes = 30

    # Create the graph and add weight
    DG = nx.erdos_renyi_graph(number_nodes, p, seed=2, directed=True)
    utils.add_weight(DG)

    # Draw the graph
    nx.draw(DG)
    plt.show()

    # Add the anomalies
    list_of_anomalies = anomalies.selection_of_anomalies()
    #anomalies.info_anomalies(list_of_anomalies)
    anomalies.insert_anomalies(DG, list_of_anomalies, w)

    # Draw it again
    nx.draw(DG)
    plt.show()

    #Do the augmentation
    #heavy_path.augmentation(DG)

    # 3.5 - Path Finder
    paths_to_consider = path_finder.paths(DG, 30)
    print(paths_to_consider)

    return 0


if __name__ == "__main__":
    main()
