####################################################################
#
#                   FEATURES BUILDER
#
#  This modules use all the modules from the paper and compute them for a network.
#  You can use it to build features on a network and store it in a pickle format.
#  This module is used to generate a lot of network with features to test the detection.
#
####################################################################

# Modules imports
import networkx as nx
from path_finder import create_features_path_finder
import numpy as np
import os
import pandas as pd
import datetime, string


# Own modules imports
from generation import generate_network
from GAW import GAW_with_null
from communities import community_feats
from localisation import localisation_feats
from net_emd import create_features_net_emd


def build_feats(G):
    ''' This function return a dataframe with all the features for each node.
        It merge all the features compute by all the modules
        Return a dataframe with index the nodes and columns the features.
    '''
    feats = pd.DataFrame(index = G.nodes())
    
    # GAW features from section 3.1
    feats_gaw = GAW_with_null(G, levels = [1, 0.1, 0.2])
    feats = feats.join(feats_gaw)
    
    # Community features of section 3.2
    feats_coms, HG_parts = community_feats(G)
    feats = feats.join(feats_coms)
    
    # Localisation features (section 3.3) using the heavy partition of seciton 3.2
    feats_locs = localisation_feats(G, HG_parts)
    feats = feats.join(feats_locs)
#    
    # NetEMD features (section 3.4)
    feats_emd = create_features_net_emd(G)
    feats = feats.join(feats_emd)
    
    # Create the feature based on path finder (3.5)
    # Real parameters are beam_width = 5000, number_monte_carlo = 500, number to keep (no idea...)
    feats_path = create_features_path_finder(G, beam_width=100, number_monte_carlo=100, number_to_keep=20)
    feats = feats.join(feats_path)

    return feats
    
    

def build_observations(w, p, n, save = True, path = ""):
    ''' Generate a network, compute its features and store it in a pickle format.
        Params:    
            w: the weight distribution
            p: the probability of an edge
            save: if the features should be save
            path: the folder where to store all features
    
    '''
    # Generate a network
    G, df_anomaly = generate_network(w,p,n)
    
    # Building features
    feats = build_feats(G)
    feats = feats.join(df_anomaly)
    
    # Save charactheristics
    feats["w"] = w
    feats["p"] = p
    feats["n"] = n
    
    # Save the features and the network
    if save:
        name = get_name(n,w,p)
        feats.to_pickle(os.path.join(path, name))
        nx.write_gpickle(G, os.path.join(path, "G_" + name))
    return feats, G


def get_name(n,w,p):
    ''' Return a appropriate name for the network.
        It will be the name of the pickle file with the features.
        This names stores the parameter of the network and the time it was generated.
    '''
    
    # Store the time in a string
    now = datetime.datetime.now()   
    t = now.strftime("%m-%d-%H-%M")
    
    name = "n{}_w{}_p{}_t{}_v2.pickle".format(n,w,p,t)
    return name
