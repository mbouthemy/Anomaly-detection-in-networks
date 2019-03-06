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



def eigen(X, k, upper = True):    
    n = X.shape[0]
    
    if k < n:
        order = "LM" if upper else "SM"
        eig_vals, eig_vecs = scipy.sparse.linalg.eigsh(X, k = k, which = order)
    else: # Too small to use sparse
        eig_vals, eig_vecs = np.linalg.eig(X.toarray())
        
        idx = eig_vals.argsort()[::-1]
        idx = np.flip(idx)
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]
        
    return eig_vals, eig_vecs


def lower_sym_eig(G, N_eigs):
    '''
        Compute the N_eigs upper eigen value of the symetrized adjacengy matrix of G
    '''
    W_s = get_sym_adj(G)
    eig_vals, eig_vecs = eigen(W_s, N_eigs, upper = False)
    return eig_vecs


def upper_sym_eig(G, N_eigs):
    '''
        Compute the N_eigs upper eigen value of the symetrized adjacengy matrix of G
    '''
    W_s = get_sym_adj(G)
    eig_vals, eig_vecs = eigen(W_s, N_eigs, upper = True)
    return eig_vecs


def upper_comb_eig(G, N_eigs):
    '''
        Compute the N_eigs upper eigen value of the symetrized adjacengy matrix of G
    '''
    L_comb, L_rw = get_laplacians(G)
    eig_vals, eig_vecs = eigen(L_comb, N_eigs + 1, upper = True)
    return eig_vecs[:, 1:]


def lower_rw_eig(G, N_eigs):
    '''
        Compute the N_eigs upper eigen value of the symetrized adjacengy matrix of G
    '''
    L_comb, L_rw = get_laplacians(G)
    eig_vals, eig_vecs = eigen(L_rw, N_eigs + 1, upper = False)
    return eig_vecs[:, :-1]


def p_val_upper(V, D):
    p_val = (1 + (np.expand_dims(V, axis=-1) <= D).sum(axis = -1))/(len(D) + 1)
    return p_val


def p_val_lower(V, D):
    p_val = (1 + (np.expand_dims(V, axis=-1) >= D).sum(axis = -1))/(len(D) + 1)
    return p_val


def get_sym_adj(G):
    W = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight="weight", format='csr')
    W_s = W + W.transpose()
    return W_s
    

def get_laplacians(G):
    W = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight="weight", format='csr')
    n,m = W.shape
    diags = W.T.sum(axis=1) + W.sum(axis = 1)
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csc')
    D_inv = scipy.sparse.spdiags(1/(diags.flatten()), [0], m, n, format='csc')
    
    W_s = W + W.transpose()
    
    L_comb = D - W_s
    L_rw = D_inv.dot(W_s)
    
    return L_comb, L_rw

    
def invert_dict(d):
    inv_map = {}
    for k, v in d.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map
