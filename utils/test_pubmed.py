import sys
import numpy as np
import _pickle as cPickle

sys.path.insert(0,'../')

import model.tensorflow_model as model
from graphs.vertex_features import vertex_features as dense

# Reading the data
features = cPickle.load(open("../datasets/pubmed/ind.pubmed.allx", 'rb'),encoding="latin1").toarray()
labels = cPickle.load(open("../datasets/pubmed/ind.pubmed.ally", 'rb'))
graph = cPickle.load(open("../datasets/pubmed/ind.pubmed.graph", 'rb'))

# Creating adjacency matrix
n = len(labels)
adj = np.zeros((n,n), dtype=int)

for i, k in graph.items():
    if i < n:
        for j in k:
            if j < n:
                adj[i][j] = 1

#-------------------------------------------------------------#

method = 2
n_lab, n_train, n_unlab = .2, .6, .2
G = dense(adj, features, labels)
num_epochs, lr = 10, 0.5

#-------------------------------------------------------------#

## Tensorflow
print("TENSORFLOW")
model1.train(H, num_epochs, lr, method)
