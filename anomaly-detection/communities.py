#
# Community features
#

import community
from utils import invert_dict
import heavy_path
import pandas as pd
import numpy as np
import networkx as nx
import scipy
from generation import generate_community_density

from GAW import GAW_G

def get_partition(G):
    partition_map = community.best_partition(G.to_undirected())
    partition = invert_dict(partition_map)

    sub_Gs = []  
    for i, nodes in partition.items():
        sub_G = G.subgraph(nodes)
        sub_Gs.append(sub_G)
        
    return sub_Gs




    

def p_value_upper(x, dist):
    return (1+sum(x >= np.array(dist)))/(len(dist) + 1)
    



def community_feats(G, density_threshold = 0.5):
    print("Compute community features :")
    print("\tAugmentation of the network...")
    HG = heavy_path.augmentation(G)
    print("\tBuilding partition...")
    HG_parts = get_partition(HG)
    
    
    # Build first feature
    features = pd.DataFrame(index = HG.nodes)
    HG_density = nx.density(HG)
    print("\tGenerate density distribution...")
    density_dist = generate_community_density(G)
    print("\tCompute full network GAW...")
    HG_GAW = pd.Series(GAW_G(HG))
    
    for i, part in enumerate(HG_parts):
        print("\tCompute community features for {}/{}...".format(i+1, len(HG_parts)))
        
        # Compute features for each community
        n_part = len(part)
        density = nx.density(part)
        density_rel = density / HG_density
        density_penalized = density_rel / n_part
        density_pvalue = p_value_upper(density, density_dist)
        score = 0 if density_pvalue >= density_threshold else scipy.stats.norm.ppf(1-density_pvalue)
        GAW_com = pd.Series(GAW_G(part))
        
        # Store features
        com_nodes = list(part.nodes)
        features.loc[com_nodes, 'com_density_rel'] = density_rel
        features.loc[com_nodes, 'com_density_penalized'] = density_penalized
        features.loc[com_nodes, 'com_density_pvalue'] = density_pvalue
        features.loc[com_nodes, 'com_score'] = score
        features.loc[com_nodes, "com_very_small"] = int(n_part < 4)
        features.loc[com_nodes, "com_GAW"] = GAW_com / HG_GAW.loc[com_nodes]
        features.loc[com_nodes, "com_GAW_penalized"] = features.loc[com_nodes, "com_GAW"] / n_part
    print("Community features have been computed !\n")
        
    return features, HG_parts





















