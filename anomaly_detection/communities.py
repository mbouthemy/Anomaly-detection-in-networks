####################################################################
#
#                   COMMUNITIES FEATURES (3.2)
#
#  This modules computes the community features of a network G
#  Use the function community_feats to get the features of your network G
#
####################################################################

# Modules imports
import community
import pandas as pd
import numpy as np
import networkx as nx
import scipy

# Own modules imports
import heavy_path
from generation import generate_community_density
from GAW import GAW_G
from utils import invert_dict



###################################
#
#    Utils for the features
#
###################################

def get_partition(G):
    """Do the partition of G with the python-louvain patckage
        Return a list of sub network of G
    """
    partition_map = community.best_partition(G.to_undirected()) # Compute the partition dictionarry
    partition = invert_dict(partition_map) # Invert the dictionary to get list of nodes for each partition

    # Build each network with the network nodes
    sub_Gs = []  
    for i, nodes in partition.items():
        sub_G = G.subgraph(nodes)
        sub_Gs.append(sub_G)
        
    return sub_Gs


def p_value_upper(x, dist):
    '''Return the upper monte carlo p-value of x according to the distribution in dist'''
    return (1+sum(x >= np.array(dist)))/(len(dist) + 1)
    



###################################
#
#    Features computation
#
###################################
    
    
def community_feats(G, density_threshold = 0.5):
    '''Compute all the communities features of section 3.2 of the paper
        Parameters :
            G: A networkx instance
            density_threshold: a p-value threshold for the density'''

    # Start by doing the partition of the augmented network
    print("Compute community features :")
    print("\tAugmentation of the network...")
    HG = heavy_path.augmentation(G)
    print("\tBuilding partition...")
    HG_parts = get_partition(HG)
    
    # Compute features for the full network
    features = pd.DataFrame(index = HG.nodes) # The container for all the features
    HG_density = nx.density(HG)
    print("\tCompute full network GAW...")
    HG_GAW = pd.Series(GAW_G(HG))
    
    # Generate a distribution of parittion density with null configuration
    print("\tGenerate density distribution...")
    density_dist = generate_community_density(G, N = 20)
    
    # Compute features for each community
    for i, part in enumerate(HG_parts):
        print("\tCompute community features for {}/{}...".format(i+1, len(HG_parts)))
        
        # Compute features for the community
        n_part = len(part) # Size of the partition
        density = nx.density(part) # Density
        density_rel = density / HG_density # Relative density
        density_penalized = density_rel / n_part # Relative density penalized
        density_pvalue = p_value_upper(density, density_dist) # Upper p-value monte carlo
        score = 0 if density_pvalue >= density_threshold else scipy.stats.norm.ppf(1-density_pvalue) # Compute the score if significant
        GAW_com = pd.Series(GAW_G(part)) # Compute the GAW of each nodes of the partition
        
        # Store features in a dataframe
        com_nodes = list(part.nodes)
        features.loc[com_nodes, 'com_density_rel'] = density_rel
        features.loc[com_nodes, 'com_density_penalized'] = density_penalized
        features.loc[com_nodes, 'com_density_pvalue'] = density_pvalue
        features.loc[com_nodes, 'com_score'] = score
        features.loc[com_nodes, "com_very_small"] = int(n_part < 4) # Special dummy for small community
        features.loc[com_nodes, "com_GAW_rel"] = GAW_com / HG_GAW.loc[com_nodes] # Relative gaw for each nodes
        features.loc[com_nodes, "com_GAW_penalized"] = features.loc[com_nodes, "com_GAW_rel"] / n_part # Penalized relative gaw
    print("Community features have been computed !\n")
        
    return features, HG_parts





















