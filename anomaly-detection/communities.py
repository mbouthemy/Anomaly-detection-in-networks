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
    



def build_community_features(G, density_threshold = 0.5):
    HG = heavy_path.augmentation(G)
    HG_parts = get_partition(HG)
    
    
    # Build first feature
    features = pd.DataFrame(index = HG.nodes)
    HG_density = nx.density(HG)
    density_dist = generate_community_density(G)
    HG_GAW = pd.Series(GAW_G(HG))
    
    for part in HG_parts:
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
        
    return features, HG_parts