import numpy as np
from graphs.vertex_features import vertex_features as graph
from graphs.vertex_features_sparse import vertex_features_sparse as sparse

import model.mxnet_model as model1
import model.tensorflow_model as model2
import model.tensorflow_sparse_model as model3
import model.pytorch_sparse_model as model4

# Reading the data
data = np.load("../datasets/cora.npz", allow_pickle=True)
adj = csr((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])
attr = csr((data['attr_data'], data['attr_indices'], data['attr_indptr']), shape=data['attr_shape'])
labels = data['labels']

# THIS WILL CHANGE IN THE FUTURE
# NOT USING SPARSE MATRICES RIGHT NOW

#-------------------------------------------------------------#

method = 2
n_lab, n_train, n_unlab = .2, .6, .2
G = graph(adj.todense(), attr.todense(), labels)
H = sparse(G)
num_epochs, lr = 10, 0.5

#-------------------------------------------------------------#

### MXNET
print("MXNET")
model1.train(G, num_epochs, lr, method)

### Tensorflow
print("\n\nTENSORFLOW")
model2.train(G, num_epochs, lr, method)

### Tensorflow sparse
print("\n\nTENSORFLOW SPARSE")
model3.train(H, num_epochs, lr, method)

### Pytorch sparse
print("\n\nPYTORCH SPARSE")
model4.train(G, num_epochs, lr, method)
