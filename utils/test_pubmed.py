import sys
import numpy as np
import _pickle as cPickle

sys.path.insert(0,'../')

import model.tensorflow_model as model

# Reading the data
features = cPickle.load(open("../datasets/pubmed/ind.pubmed.allx", 'rb'),encoding="latin1").toarray()
labels = cPickle.load(open("../datasets/pubmed/ind.pubmed.ally", 'rb'))
graph = cPickle.load(open("../datasets/pubmed/ind.pubmed.graph", 'rb'))

# Creating adjacency matrix
n = len(graph)
adj = np.zeros((n,n), dtype=int)

for i, k in graph.items():
    for j in k:
        adj[i][j] = 1

#-------------------------------------------------------------#

method = 2
n_lab, n_train, n_unlab = .2, .6, .2
G = model(adj, attr, labels)
num_epochs, lr = 10, 0.5

#-------------------------------------------------------------#

## Tensorflow
print("TENSORFLOW")
model1.train(H, num_epochs, lr, method)
