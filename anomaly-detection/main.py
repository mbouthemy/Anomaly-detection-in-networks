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


from GAW import GAW_with_null

        


w = 0.7
p = 0.05
number_nodes = 500

G = nx.erdos_renyi_graph(number_nodes, p, directed=True)
utils.add_weight(G)


list_of_anomalies = anomalies.selection_of_anomalies()
#anomalies.info_anomalies(list_of_anomalies)
anomalies.insert_anomalies(G, list_of_anomalies, w)    




feats_gaw = GAW_with_null(G, levels = [1, 0.1, 0.2])
feats_com, HG_parts = build_community_features(G)



loc_feats = pd.DataFrame()
for i, part in enumerate(HG_parts):
    print("Compute {}/{}...".format(i+1, len(HG_parts)))
    res = compute_eigen_features(part, eig_generator = upper_eig_generator, N_eigs = 20, N_null = 500)
    loc_feats = loc_feats.append(res)


