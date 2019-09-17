import sys
import numpy as np
import _pikcle as cPickle

sys.path.insert(0,'../')

import model.tensorflow_model as model

# Reading the data
features = cPickle.load(open("datasets/pubmed/ind.pubmed.allx"))
labels = cPickle.load(open("datasets/pubmed/ind.pubmed.ally"))
graph = cPickle.load(open("datasets/pubmed/ind.pubmed.graph"))

n = labels.shape[0]
adj = [[0]*n]*n

for i in graph:
    for j in graph[i]:
        adj[i][j] = 1

#-------------------------------------------------------------#

method = 2
n_lab, n_train, n_unlab = .2, .6, .2
G = sparse(adj, attr, labels)
num_epochs, lr = 10, 0.5

#-------------------------------------------------------------#

## Tensorflow
print("TENSORFLOW")
model1.train(H, num_epochs, lr, method)
