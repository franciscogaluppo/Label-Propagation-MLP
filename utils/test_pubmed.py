import sys
import numpy as np
import _pickle as cPickle
from scipy.sparse import csr_matrix as csr

sys.path.insert(0,'../')

import model.tensorflow_sparse_model as model1
import model.pytorch_sparse_model as model2
from graphs.vertex_features_csr import vertex_features_csr as sparse

# Reading the data
attr = cPickle.load(open("../datasets/pubmed/ind.pubmed.allx", 'rb'),encoding="latin1")
labels = cPickle.load(open("../datasets/pubmed/ind.pubmed.ally", 'rb'))
graph = cPickle.load(open("../datasets/pubmed/ind.pubmed.graph", 'rb'))

# Creating adjacency matrix
n = len(labels)

row = list()
col = list()
data = list()

for i, k in graph.items():
    if i < n:
        for j in np.unique(k):
            if j < n and j != i:
                row.append(i)
                col.append(j)
                data.append(1)

adj = csr((data, (row, col)), shape=(n, n))

#-------------------------------------------------------------#

method = 2
n_lab, n_train, n_unlab = .2, .6, .2
G = sparse(adj, attr, labels, one_hot=True)
num_epochs, lr = 10, 0.5

#-------------------------------------------------------------#

## Tensorflow
print("TENSORFLOW")
#model1.train(G, num_epochs, lr, method)

## Pytorch
print("\n\nPYTORCH")
model2.train(G, num_epochs, lr, method)
