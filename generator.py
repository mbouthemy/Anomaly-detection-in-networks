####################################################################
#
#                   GENERATOR OF NETWORK FEATURES
#
#  This files is used to generate a lot of network features.
#  It stores them in the features folder.
#  It will only ask you the number of nodes for each network.
#  Then it will generate network with the parameters as in the paper.
#  For each network it computes its features and store it.
#
####################################################################


# Module imports
import numpy as np
import pandas as pd
import os
import random, time

# The folder of this files -> usefull to get the features folder
dir_path = os.path.dirname(os.path.realpath(__file__))

# Loading the library to python 
import sys
sys.path.insert(0, os.path.join(dir_path, "anomaly_detection/"))



# Own import
from anomaly_detection.features import build_observations


def parameters_range_theoric(n, k = 5):
    '''
         The theoritical parameters range from the paper but does not work has p is too small
    '''
    lim = np.prod(np.arange(n-k+1, n+1)**(-1/(k-1)))
    res = []
    p = 1/1000
    while p <= 1/200: 
        w = 0.99
        while(w <= 1):
            if (1-w) * p < lim:
                res.append([w,p])
            w += 1/1000
        p += 1/1000
    res = np.array(res)
    return res


def parameters_range():
    '''
        Fixed range to generate the model with p bigger enough
    '''
    res = []
    for p in np.arange(0.01, 0.06, 0.01):
        for w in np.arange(0.4, 1.1, 0.1):
            res.append((w,p))
    return np.array(res)
  

def generate_observations(n, params, N = 1000, path = "features"):
    ''' Generator of networks.
        Generates a network, computes its features and stores it in path
        
        Params:
            n: the number of nodes
            params: the set of parameters to use for the generation
            N: the number of generation
            path: the folder to store the features'''            
        
    for i_run in range(N):
        w, p = random.choice(params) # pick a random set of parameters
        t = time.perf_counter()
        print("\n\n####  Network n°{} : Running for (w,p,n) = {}  ####\n\n".format(i_run+1, (w,p,n)))
        try:
            build_observations(w = w, p = p, n = n, path = path, save = True)
        except KeyboardInterrupt:
            print("\n\n#### GENERATION HAS BEEN ABORTED BY THE USER ! ####\n\n")
            return False
        except Exception as e:
            time_spent = round(time.perf_counter() - t) # in seconds
            print("\n\n#### ERROR #### The generation for (w,p,n) = {} has crashed ! ({} seconds)  #### ERROR ####".format((w,p,n), time_spent))
            print("#### ERROR #### {}  #### ERROR ####\n\n".format(e))
        else:
            time_spent = round(time.perf_counter() - t) # in seconds
            print("\n\n####  Features for (w,p,n) = {} have been saved ! ({} seconds)  ####\n\n".format((w,p,n), time_spent))



def main():
    path_feats = os.path.join(dir_path, "features/")
    print("Path for features {}".format(path_feats))
    
    n = int(input("Number of nodes for the generation : "))
    
    paper = True
    params = parameters_range_theoric(n) if paper else parameters_range()
    generate_observations(n = n, params = params, path = path_feats)


if __name__ == "__main__":
    main()

