# Perform the augmentation and the community detection with the Python Louvain algorithm.
import numpy as np


def weight_distribution_percentile(matrix):
    """Get the 99th-percentile of the weight distribution over the graph."""
    weight_list = []
    list_nodes = matrix.flatten()  # Transform the matrix to a list.
    for x in list_nodes:
        if x != float(0):
            weight_list.append(x)
    threshold = np.percentile(weight_list, 99)
    return threshold


def augmentation_matrix(matrix):
    """Find heavy path in the graph and augment those path."""
    new_matrix = matrix.copy()
    threshold = weight_distribution_percentile(matrix)
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if new_matrix[i, j] >= threshold:  # First Edge.
                for k in range(n):
                    if new_matrix[j, k] >= threshold and k != i:  # Second Edge possible.

                        new_matrix[i, k] = max(new_matrix[i, k], min(new_matrix[i, j], new_matrix[j, k]))
                        # Update is min(w_3, max(w_1, w_2))

    return new_matrix

