import sys
import numpy as np
from scipy.sparse import csr_matrix as csr

sys.path.insert(0,'../')

from graphs.vertex_features_csr import vertex_features_csr as sparse
from graphs.csr2dense import csr2dense as dense
import model.tensorflow_model as model1
import model.tensorflow_sparse_model as model2

# Reading the data
data = np.load("../datasets/cora_ml.npz", allow_pickle=True)
adj = csr((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])
attr = csr((data['attr_data'], data['attr_indices'], data['attr_indptr']), shape=data['attr_shape'])
labels = data['labels']

# THIS WILL CHANGE IN THE FUTURE
# NOT USING SPARSE MATRICES RIGHT NOW

#-------------------------------------------------------------#

method = 2
n_lab, n_train, n_unlab = .2, .6, .2
G = sparse(adj, attr, labels)
H = dense(G, adj.toarray())
num_epochs, lr = 10, 0.5

#-------------------------------------------------------------#

## Tensorflow
print("TENSORFLOW")
model1.train(H, num_epochs, lr, method)

### Tensorflow sparse
print("\n\nTENSORFLOW SPARSE")
#model.train(G, num_epochs, lr, method)
