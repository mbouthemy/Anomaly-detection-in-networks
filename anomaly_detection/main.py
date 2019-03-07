import networkx as nx
import anomalies
from path_finder import create_features_path_finder
import utils


def main():
    w = 0.7
    p = 0.05
    number_nodes = 50

    G = nx.erdos_renyi_graph(number_nodes, p, seed=2, directed=True)
    utils.add_weight(G)

    list_of_anomalies = anomalies.selection_of_anomalies()
    anomalies.info_anomalies(list_of_anomalies)
    df_anomaly = anomalies.insert_anomalies(G, list_of_anomalies, w)

    """
    # Global features container
    features = pd.DataFrame()
    
    # Compute community feature
    com_features, HG_parts = build_community_features(G)
    features = com_features  # Add the
    
    
    loc_feats = pd.DataFrame()
    for i, part in enumerate(HG_parts):
        print("Compute {}/{}...".format(i+1, len(HG_parts)))
        res = compute_eigen_features(part, eig_generator = upper_eig_generator, N_eigs = 20, N_null = 500)
        loc_feats = loc_feats.append(res)
    features = features.join(loc_feats)  # Add localisation features
    """

    # Create the feature based on path finder (3.5)
    # Real parameters are beam_width = 5000, number_monte_carlo = 500, number to keep (no idea...)
    df_features_path_finder = create_features_path_finder(G, beam_width=100, number_monte_carlo=100, number_to_keep=20)


if __name__ == '__main__':
    main()
