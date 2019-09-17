from graphs.graph import graph
import numpy as np
from operator import itemgetter as get
from itertools import groupby as gb
from scipy import sparse

class csr2dense(graph):

    def __init__(self, G, adj):
        """
        Class initialization
        :param G:
        :param adj: 
        """    
        
        self.vertices = G.vertices
        self.n_lab = G.n_lab

        self.labels = G.labels
        self.n_unlab = G.n_unlab
        self.n_train = G.n_train
        self.n_feat = G.n_feat
        self.Ytest = G.Ytest
        self.Ylabel = G.Ylabel
        self.Ytarget = G.Ytarget

        self.adj = adj
        self.feats = sparse.coo_matrix((G.feats_sparse[1],
            tuple(zip(*G.feats_sparse[0]))), shape=G.feats_sparse[2]).toarray()

