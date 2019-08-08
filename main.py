import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from graphs.random_features import random_features as graph
import model.mxnet_model as model1
import model.tensorflow_model as model2

#-------------------------------------------------------------#

# Cria um grafo exemplo
method, labels = 2, 2
n_lab, n_train, n_unlab, n_feat, p = 20, 60, 20, 10, 0.3
G = graph(n_lab, n_train, n_unlab, n_feat, labels, method, p=p)
num_epochs, lr = 20, 0.5

#-------------------------------------------------------------#

### MXNET
print("MXNET")
model1.train(G, num_epochs, lr, method)

### Tensorflow
print("\n\nTENSORFLOW")
model2.train(G, num_epochs, lr, method)
