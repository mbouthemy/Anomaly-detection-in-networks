import networkx as nx
import anomalies
from path_finder import create_features_path_finder
import utils
import os

from GAW import GAW_with_null

from communities import community_feats

from localisation import localisation_feats

import pandas as pd

import datetime, random, string

def build_feats(G):
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
    
    # Create the feature based on path finder (3.5)
    # Real parameters are beam_width = 5000, number_monte_carlo = 500, number to keep (no idea...)
    feats_path = create_features_path_finder(G, beam_width=100, number_monte_carlo=100, number_to_keep=20)
    feats = feats.join(feats_path)

    return feats

def generate_network(w, p, n):
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
    
    

def build_observations(w, p, n, save = True, path = ""):
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
    # Store the time in a string
    now = datetime.datetime.now()   
    t = now.strftime("%m-%d-%H-%M")
    
    # Generate a random string for a unique name
    has = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    
    name = "n{}_w{}_p{}_t{}_{}.pickle".format(n,w,p,t,has)
    return name