from graphs.graph import graph
import numpy as np
from operator import itemgetter as get
from itertools import groupby as gb

class vertex_features_csr(graph):

    def __init__(self, adj, features, labels, n_lab=0.2, n_train=0.6, n_unlab=0.2):
        """
        Class initialization
        :param adj: 
        :param attr: 
        :param labels:
        :param n_lab:
        :param n_train:
        :param n_unlab:
        """    

        # Salva valores para que os outros métodos possam acessar
        n = self.vertices = int(adj.shape[0])
        l = self.n_lab = int(np.floor(n_lab*n))
        k = n-l

        self.n_train = int(np.ceil(n_train*n))
        self.n_unlab = n - (self.n_lab + self.n_train)

        # Salva adj e a lista de arestas
        self.adj = adj
        self.edges = int (np.sum(adj))
        self.edge_list = [(i, j) for i in range(n)
                                 for j in range(n)
                                 if self.adj[(i,j)]]

        # Chama os métodos de criação
        self.features(features)

        # Guarda os valores
        self.Ylabel = labels[:self.n_lab,:] 
        self.Ytarget = labels[self.n_lab:(self.n_train+self.n_lab),:]
        self.Ytest = labels[-self.n_unlab:,:]


    def features(self, attr):
        """
        Vertex features pooling
        """

        n = self.vertices
        l = self.n_lab
        k = n-l

        self.n_feat = 2 * attr.shape[1]
        feats = np.zeros((n*n, self.n_feat))
        for e in range(self.edges):
            i, j = self.edge_list[e]
            feats[i*n+j] = np.concat((attr[]))
        self.feats = feats.T
