from graphs.graph import graph
import numpy as np
from operator import itemgetter as get
from itertools import groupby as gb
from scipy.sparse import csr_matrix as csr
from scipy.sparse import coo_matrix as coo

def sparse_diag(a):
    m = a.shape[1]
    dim = np.prod(a.shape)
    coo = a.tocoo()
    indices = [[b[0]*m + b[1]]*2 for b in np.mat([coo.row, coo.col]).transpose().tolist()]
    return (indices, coo.data, (dim, dim))

class vertex_features_csr(graph):

    def __init__(self, adj, attr, labels, n_lab=0.2, n_train=0.6, n_unlab=0.2, one_hot=False):
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
        self.adj_sparse = sparse_diag(adj[-k:,].reshape((k*n, 1)).astype(np.float64))
        self.edges = int (np.sum(adj))
        adj_coo = adj.tocoo()
        self.edge_list = [tuple(x) for x in
            sorted(np.vstack((adj_coo.row, adj_coo.col)).transpose().tolist())]

        # Chama os métodos de criação
        self.features(attr)
        self.arrays(labels, one_hot)


    def features(self, attr):
        """
        Vertex features pooling
        """

        n = self.vertices
        l = self.n_lab
        k = n-l

        self.n_feat = 2 * attr.shape[1]
        
        indices = list()
        data = list()
        shape = (self.n_feat, k*n)
        
        for e in range(self.edges):
            i, j = self.edge_list[e]

            # Precisamnos apenas das arestas que saem dos últimos k vértices
            if i >= k:
                a = attr[i].tocoo()
                b = attr[j].tocoo()

                # Adiciona os elementos do primeiro vértice
                for p in range(len(a.row)):
                    indices.append([p, (i-k)*n+j])
                    data.append(a.data[p])
    
                # Adiciona os elementos do segundo vértice
                for q in range(len(b.row)):
                    indices.append([q+attr.shape[1], (i-k)*n+j])
                    data.append(b.data[q])
    
        self.feats_sparse = coo((data, tuple(zip(*indices))), shape=shape)


    def arrays(self, data_labels, one_hot):
        """
        Split vertices between labeled, train and unlabeled
        """

        # Cria os 1-hot encoded labels 
        if not one_hot:    
            self.labels = max(data_labels)+1
            Ylabel = np.zeros((self.vertices, self.labels))
            Ylabel[np.arange(self.vertices), data_labels] = 1

        else:
            self.labels = data_labels.shape[1]
            Ylabel = data_labels

        # Guarda os valores
        self.Ylabel = Ylabel[:self.n_lab,:] 
        self.Ytarget = Ylabel[self.n_lab:(self.n_train+self.n_lab),:]
        self.Ytest = Ylabel[-self.n_unlab:,:]
