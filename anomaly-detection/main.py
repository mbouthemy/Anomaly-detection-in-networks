import numpy as np
import pandas as pd
import scipy.stats
import random
import math
import community
import networkx as nx
import matplotlib.pyplot as plt



import anomalies
import path_finder
from communities import get_partition, build_community_features
from localisation import compute_eigen_features
from utils import upper_eig_generator
import utils


w = 0.7
p = 0.05
number_nodes = 1000

G = nx.erdos_renyi_graph(number_nodes, p, seed=2, directed=True)
utils.add_weight(G)


list_of_anomalies = anomalies.selection_of_anomalies()
anomalies.info_anomalies(list_of_anomalies)
anomalies.insert_anomalies(G, list_of_anomalies, w)


# Global features container
features = pd.DataFrame()

# Compute community feature
com_features, HG_parts = build_community_features(G)
features = com_features # Add the 


loc_feats = pd.DataFrame()
for i, part in enumerate(HG_parts):
    print("Compute {}/{}...".format(i+1, len(HG_parts)))
    res = compute_eigen_features(part, eig_generator = upper_eig_generator, N_eigs = 20, N_null = 500)
    loc_feats = loc_feats.append(res)
features = features.join(loc_feats) # Add localisation features


# 3.5 - Path Finder
paths_to_consider = path_finder.paths(DG, 30)
print(paths_to_consider)
