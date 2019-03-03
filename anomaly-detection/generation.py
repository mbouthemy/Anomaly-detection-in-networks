#
# Generation of distribution with null configuration
#

import pandas as pd
import numpy as np
import networkx as nx
import random
import heavy_path
import community
from utils import invert_dict



def generate_null(G):
    s_in, s_out, W = zip(*G.edges(data = "weight"))
    s_in, s_out, W = list(s_in), list(s_out), list(W)
    
    random.shuffle(s_in)
    random.shuffle(s_out)
    random.shuffle(W)
    
    D = nx.DiGraph()
    D.add_weighted_edges_from(zip(s_in, s_out, W))
    D.remove_edges_from(D.selfloop_edges())
    
    return D


def generate_null_with(G, N_eigs, N_tries = 10):
    for i in range(N_tries):
        G_null = generate_null(G)
        if len(G_null) >= N_eigs:
            return G_null
    if len(G_null) >= 2:
        return G_null
    else:
        raise Exception("The generation of the null replica failed after {} tries !".format(N_tries))
        
        
def generate_null_distrib(G, eig_generator, stats, N_null, N_eigs):
    T = {} # Container for all statistics
    # Generate N_null configuration model and run store each stats
    for i_null in range(N_null):
        G_null = generate_null_with(G, N_eigs=N_eigs) # Generate a null
        
        eigs = eig_generator(G_null, N_eigs) # Compute its eigs
        
        # Compute each stat and append it to the previous one
        for name, stat in stats.items():
            T_stat = stat(eigs)
            if name in T:
                T[name] += list(T_stat)
            else:
                T[name] = list(T_stat)
    
    # Convert each list to numpy array for convenience
    for name, x in T.items():
        T[name] = np.array(x)
    
    return T



def generate_community_density(G, N = 20):
    res = np.zeros(N)
    for i in range(N):
        # Generate and augmente a null
        G_null = generate_null(G)
        HG_null = heavy_path.augmentation(G_null)
        
        # Do the partition of the null
        partition_map = community.best_partition(HG_null.to_undirected())
        partition = invert_dict(partition_map)
        
        # Choose a community and compute its density
        G_community = G_null.subgraph(random.choice(partition))
        d = nx.density(G_community)
        res[i] = d
        
    return res