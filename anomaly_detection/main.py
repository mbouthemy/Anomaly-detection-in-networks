import networkx as nx
import anomalies
from path_finder import create_features_path_finder
import utils


from features import build_observations

df, G = build_observations(0.01, 0.01, 50)

import pandas as pd
df = pd.read_pickle("../features/n1000_w0.7_p0.01_t03-12-16-03.pickle")