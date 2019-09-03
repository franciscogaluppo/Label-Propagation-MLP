from graphs.graph import graph
import numpy as np
from operator import itemgetter as get
from itertools import groupby as gb

direct = lambda x: x if x[0] <= x[1] else x[::-1]
direct_list = lambda x: [direct(a) for a in x]

class vertex_features(graph):

    def __init__(self, adj, attr, labels, n_lab=0.2, n_train=0.6, n_unlab=0.2):
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
        n = self.vertices = adj.shape[0]
        self.n_lab = np.floor(n_lab*n)
        self.n_train = np.ceil(n_train*n)
        self.n_unlab = n - (self.n_lab + self.n_train)
        self.n_feat = ...

        # Chama os métodos de criação
        self.features()
        self.create_arrays()


    def features(self):
        n = self.vertices


    def arrays(self):
        pass
