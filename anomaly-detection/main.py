import networkx as nx
import anomalies
from path_finder import create_features_path_finder
import utils


from GAW import GAW_with_null

from communities import community_feats

from localisation import localisation_feats



def main():
    w = 0.7
    p = 0.05
    number_nodes = 500

    G = nx.erdos_renyi_graph(number_nodes, p, seed=2, directed=True)
    utils.add_weight(G)

    list_of_anomalies = anomalies.selection_of_anomalies()
    anomalies.info_anomalies(list_of_anomalies)
    df_anomaly = anomalies.insert_anomalies(G, list_of_anomalies, w)
    
    
    feats_gaw = GAW_with_null(G, levels = [1, 0.1, 0.2]) # 3.1

    feats_coms, HG_parts = community_feats(G) # 3.2
    
    feats_locs = localisation_feats(G, HG_parts) # 3.3

    # Create the feature based on path finder (3.5)
    # Real parameters are beam_width = 5000, number_monte_carlo = 500, number to keep (no idea...)
    feats_path = create_features_path_finder(G, beam_width=100, number_monte_carlo=100, number_to_keep=20)


if __name__ == '__main__':
    main()
