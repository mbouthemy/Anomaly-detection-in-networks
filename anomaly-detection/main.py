import numpy as np
import random
from scipy import special
import math
from utils import compute_product
from graph import Graph
from basic_detection import get_score_nodes


def main():
    n = 30
    p = 0.5
    weight = 0.7
    graph = Graph(n, p, weight)
    graph.create_graph()
    graph.selection_of_anomalies()
    graph.info_anomalies()
    graph.insert_anomalies()

    score_nodes = get_score_nodes(graph.matrix, number_monte_carlo=50)
    graph.show_graph()

    print("ta mere")
    return 0











if __name__ == "__main__":
    main()
