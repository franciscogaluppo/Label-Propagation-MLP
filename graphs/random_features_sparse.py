import numpy as np
from scipy import sparse as sp
from graphs.random_features import random_features

def sparse(a):
    coo = sp.csr_matrix(a).tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return (indices, coo.data, coo.shape)

def sparse_diag(a):
    m = a.shape[1]
    dim = np.prod(a.shape)
    coo = sp.csr_matrix(a).tocoo()
    indices = [[b[0]*m + b[1]]*2 for b in np.mat([coo.row, coo.col]).transpose().tolist()]
    return (indices, coo.data, (dim, dim))

class random_features_sparse(random_features):
    def __init__(self, G):
        n = self.vertices = G.vertices
        l = self.n_lab = G.n_lab
        k = n-l

        self.labels = G.labels
        self.n_unlab = G.n_unlab
        self.n_train = G.n_train
        self.n_feat = G.n_feat
        self.Ytest = G.Ytest
        self.Ylabel = G.Ylabel
        self.Ytarget = G.Ytarget
        
        self.adj_sparse = sparse_diag(G.adj[-k:,].reshape((k*n, 1)).astype(np.float64))
        self.feats_sparse = sparse(G.feats[:,-k*n:])
