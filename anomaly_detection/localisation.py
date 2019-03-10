"""
    Localisation features
"""

import numpy as np
import pandas as pd
import scipy
from math import sqrt
import networkx as nx
import community
from generation import generate_null_distrib

from utils import p_val_upper, upper_eig_generator

#
# Statistic function for the test of eigen value
#
    
def norm_IPR(x):
    '''
        Compute the IPR for every element of x
        Warning : To have the norm you need to do the sum after !
    '''
    return x**4

def norm_exp(x):
    '''
        Compute the exponential formula for every element of x
        Warning : To have the norm you need to do the sum after !
    '''
    x_abs = np.abs(x)
    return (np.exp(x_abs) - x_abs - 1)    


def sign_based_test(eig_vecs):
    N_pos = np.sum(eig_vecs > 0, axis = 0)
    N_neg = np.sum(eig_vecs < 0, axis = 0)
    teta_G = min(20, len(eig_vecs)) # do not handle trivial yet !
    
    T = np.minimum(N_pos + (N_pos == 0).astype(np.int) * teta_G, N_neg + (N_pos == 0).astype(np.int) * teta_G)/teta_G
    return T





#
# Features computation
#



def sign_based_features(D, eigs, threshold = 0.05): 
    '''
        Compute the sign based features of the paper
        
        D    : contains a monte-carlo distribution of the sign based test
        eigs : the eigens value for which to compute the features
        
        Return the four features of the paper (formula 21, 22, 23, 24)
    '''    
    # Compute the p_val for each vector
    V = sign_based_test(eigs)
    p_val = p_val_upper(V, D)
    
    # Only keep significant vector
    vector_selector = (p_val < threshold)
    n = sum(vector_selector)
    if n == 0: # Nothing significant
        return np.zeros((4, len(eigs)))
    eigs = eigs[:, vector_selector]
        
    N_pos = np.sum(eigs > 0, axis = 0)
    N_neg = np.sum(eigs < 0, axis = 0)
    
    N_pos = N_pos + (N_pos == 0).astype(np.int) * N_pos.shape[0]
    N_neg = N_neg + (N_neg == 0).astype(np.int) * N_neg.shape[0]
    
    cdf = scipy.stats.norm.cdf(1-p_val)
    
    # Decomposition of the below formula (save computation)
    a = (N_pos < N_neg).astype(np.int) * cdf * (eigs > 0).astype(np.int)
    b = (N_pos > N_neg).astype(np.int) * cdf * (eigs < 0).astype(np.int)
    
    Sign_Stat_1 = a + b # formula (21)
    Sign_Stat_2 = a/N_pos + b/N_neg # formula (22)
    
    # formula (21) and (22)
    Sign_Stat_Equal_1 = (N_pos == N_neg).astype(np.int) * (eigs > 0).astype(np.int) + cdf * (eigs < 0).astype(np.int)
    Sign_Stat_Equal_2 = Sign_Stat_Equal_1  / (N_pos + N_neg)
    
    return Sign_Stat_1.sum(axis = -1), Sign_Stat_2.sum(axis = -1), Sign_Stat_Equal_1.sum(axis = -1), Sign_Stat_Equal_2.sum(axis = -1)








def norm_based_features(norm, D, S, eigs, threshold = 0.05):    
    '''
        Compute the norm based features of the paper
        
        norm      : element-wise function that compute the norm (it does not do the sum)
        D         : contains a monte-carlo distribution of the norm
        S         : contains a monte-carlo distribution of the abolute value of eigen value
        eigs      : the eigens value for which to compute the features
        threshold : the p value threshold for the eigen vector test
        
        Return the four features of the paper (formula 17, 18, 19, 20)
    '''
    n_node = eigs.shape[0]
    
    # Select vector for norm based
    V_i = norm(eigs)
    V = np.sum(V_i, axis = 0)
    p_val = p_val_upper(V, D)
    
    # Only keep significant vector
    vector_selector = (p_val < threshold)
    n = sum(vector_selector)
    if n == 0: # Nothing significant
        return np.zeros((4, len(eigs)))
    eigs = eigs[:, vector_selector]
    p_val = p_val[vector_selector]
    V_i = V_i[:, vector_selector]
    
    # Compute g_1 (17)
    g_1 = np.abs(V_i)/sqrt(n_node)
    
    
    # Compute g_2 (18)
    t_a = p_val_upper(np.abs(eigs), S) # compute t(a)
    g_2 = np.zeros(eigs.shape) # initialize at zero
    idx_signi = np.where(t_a < 0.5) # get the position of significant value
    g_2[idx_signi] = scipy.stats.norm.cdf(1 - t_a[idx_signi]) # Update significant with CDF
    
    
    # Compute g_3 (19)
    vec_stat = np.maximum(np.expand_dims(scipy.stats.norm.cdf(1-p_val), axis=-1), 0)
    g_3 = np.tile(vec_stat, n_node).T
    idx_not_signi = np.where(np.abs(eigs) < S.mean())
    g_3[idx_not_signi] = 0
    
    
    # Compute g_4 (20)
    g_4 = np.abs(V_i)
    g_4[idx_not_signi] = 0
    
    return g_1.sum(axis = -1), g_2.sum(axis = -1), g_3.sum(axis = -1), g_4.sum(axis = -1)






def compute_eigen_features(G, eig_generator, N_eigs, N_null):
    stats = {
            "abs_max" : lambda x : np.max(np.abs(x), axis = 0),
            "pow_4_sum": lambda x : np.sum(norm_IPR(x), axis = 0),
            "exp_sum" : lambda x : np.sum(norm_exp(x), axis = 0),
            "sign_based" : sign_based_test
            
        }
    
    D = generate_null_distrib(G, eig_generator = upper_eig_generator, stats = stats, N_null = N_null, N_eigs = N_eigs)
    eigs = upper_eig_generator(G, N_eigs)
    
    feats = {}
    
    # Norm based features for the two norms
    norms = {"IPR" : norm_IPR, "EXP" : norm_exp}
    for name, norm in norms.items():
        g = norm_based_features(norm, D["pow_4_sum"], D["abs_max"], eigs)
        feats[name + "_g1"], feats[name + "_g2"], feats[name + "_g3"], feats[name + "_g4"] = g
    
    
    # Sign based
    s = sign_based_features(D["sign_based"], eigs)
    feats["SignStat1"], feats["SignStat2"], feats["SignStatEqual1"], feats["SignStatEqual2"] = s
    
    
    # Format the features to a dataframe
    feats = pd.DataFrame.from_dict(feats)
    feats.index = list(G.nodes)
    return feats
    