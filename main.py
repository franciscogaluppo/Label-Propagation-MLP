import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from graphs.random_features import random_features as graph
from graphs.random_features_sparse import random_features_sparse as sparse
import model.mxnet_model as model1
import model.tensorflow_model as model2
import model.tensorflow_sparse_model as model3
import model.pytorch_sparse_model as model4


#-------------------------------------------------------------#

# Cria um grafo exemplo
method, labels = 2, 10
n_lab, n_train, n_unlab, n_feat, p = 500, 250, 250, 10, 0.3
G = graph(n_lab, n_train, n_unlab, n_feat, labels, method, p=p)
H = sparse(G)
num_epochs, lr = 10, 0.5

#-------------------------------------------------------------#

# ### MXNET
# print("MXNET")
# model1.train(G, num_epochs, lr, method)

# ### Tensorflow
# print("\n\nTENSORFLOW")
# model2.train(G, num_epochs, lr, method)

# ### Tensorflow sparse
# print("\n\nTENSORFLOW SPARSE")
# model3.train(H, num_epochs, lr, method)

### Pytorch sparse
print("\n\nPYTORCH SPARSE")
model4.train(G, num_epochs, lr, method)
