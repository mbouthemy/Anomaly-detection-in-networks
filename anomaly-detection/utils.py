# utils.py
import numpy as np
import networkx as nx
import scipy


def add_weight(direct_graph):
    """Add weight to the existing Erdos Renyi Graph."""
    for e in direct_graph.edges():
        direct_graph[e[0]][e[1]]['weight'] = np.random.uniform()


def compute_product(list_of_elements, begin):
    """Compute the product of a list"""
    product = 1
    for j in range(begin, len(list_of_elements)):
        product *= list_of_elements[j]
    return product






def upper_eig_generator(G, N_eigs):
    '''
        Compute the N_eigs upper eigen value of the symetrized adjacengy matrix of G
    '''
    W_s = get_sym_adj(G)
    n = len(G)
    N_eigs = min(N_eigs, n)
    if N_eigs < 0.8*n:
        eig_vals, eig_vecs = scipy.sparse.linalg.eigsh(W_s, k = N_eigs, which = "LM")
    else: # Too small to use sparse
        eig_vals, eig_vecs = scipy.linalg.eigh(W_s.toarray(), k = N_eigs, which = "LM")
    return eig_vecs


def p_val_upper(V, T):
    p_val = (1 + (np.expand_dims(V, axis=-1) <= T).sum(axis = -1))/(len(T) + 1)
    return p_val


def get_sym_adj(G):
    W = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight="weight", format='csr')
    W_s = W + W.transpose()
    return W_s
    

def get_laplacians(G):
    W = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight="weight", format='csr')
    n,m = W.shape
    diags = W.sum(axis=1)
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csc')
    
    W_s = W + W.transpose()
    
    L_comb = D - W_s
    L_rw = scipy.sparse.linalg.inv(D).dot(W_s)
    
    return L_comb, L_rw

    
def invert_dict(d):
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map
