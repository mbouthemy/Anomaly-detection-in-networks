####################################################################
#
#                   GENERATION OF NETWORKS
#
#  This modules is used to first generate test network (ER) and configuration model from a network G.
#  Moreover, for a given a statistics you can compute its distribution across a set of configuration models.
#
####################################################################

# Modules imports
import pandas as pd
import numpy as np
import networkx as nx
import random
import community


# Own modules imports
import heavy_path
from utils import invert_dict



###################################
#
#    Network generation
#
###################################


def generate_network(w, p, n):
    ''' Generation of a network with anomalies like in the paper
        Return a networkx instance and a dataframe saying if a nodes is an anomaly
    '''
    print("Generating a network...")
    G = nx.erdos_renyi_graph(n, p, directed=True)
    utils.add_weight(G)
    list_of_anomalies = anomalies.selection_of_anomalies()
    df_anomaly = anomalies.insert_anomalies(G, list_of_anomalies, w)
    
    # Remove isolates
    isol = list(nx.isolates(G))
    if len(isol) > 0:
        print("Warning {} nodes were removed because isolated !".format(len(isol)))
        G.remove_nodes_from(isol)
        df_anomaly = df_anomaly.loc[list(G.nodes)].copy()
    print("A network has been generated.\n")
    
    return G, df_anomaly

def generate_null(G):
    ''' Generate a configuration models of G with the same weight
        It is done like in the paper.
        Return a generated configuration model
    '''
    # Get the in-stubs, out-stubs and weight list
    s_in, s_out, W = zip(*G.edges(data = "weight"))
    s_in, s_out, W = list(s_in), list(s_out), list(W)
    
    # Shuffle everything
    random.shuffle(s_in)
    random.shuffle(s_out)
    random.shuffle(W)
    
    # Build a graph with the shuffled list
    D = nx.DiGraph()
    D.add_weighted_edges_from(zip(s_in, s_out, W)) # Add the shuffle edge
    D.remove_edges_from(D.selfloop_edges()) # Remove self loop
    D.remove_nodes_from(list(nx.isolates(D))) # Remove isolated nodes
    
    return D


def generate_null_with(G, N_eigs, N_tries = 10):
    ''' Robust generation of a configuration model.
        It handles that sometimes the configuration model can be too small
        If the network is too small for the number of eigen value wanted it tries again.
        Return a networkx instance (configuration model)
        Params:
            G : a networkX instance
            N_eigs : number of eigenvalue wanted for the configuration model
            N_tries : Number of tries for the generation'''

    for i in range(N_tries):
        G_null = generate_null(G)
        if len(G_null) >= N_eigs: # Condition on the eigenvector number
            return G_null
    if len(G_null) >= 2:
        return G_null
    else:
        raise Exception("The generation of the null replica failed after {} tries !".format(N_tries))
       
        
        

###################################
#
#    Generation of Monte-Carlo distribution
#
###################################        
        
        
def generate_null_distrib(G, eig_generator, stats, N_null, N_eigs):
    ''' A utilities to generate distribution for a set of statistics on the eigen value
        Params :
            G : a networkX instance
            eig_generator : a function to generate the eigen value of the network
            stats : a dictionary of each statistics (index : stat_name ; value : function that takes eigen vectors and compute a statistics)
            N_null : Number of sample in the distribution
            N_eigs : number of minimum eigen value for the generator
    
    '''
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
    ''' A utilities to compute the distribution of the density in the heavy communities
        Params:
            G : a networkX instance
            N : the number of sample in the distribution
        Return the MC distribution of the density of a community in G
    '''
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