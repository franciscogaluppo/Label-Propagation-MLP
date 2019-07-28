from graphs.random_features import random_features as graph

# Cria um grafo exemplo
method = 2
n_lab, n_train, n_unlab, n_feat, p = 200, 1000, 100, 30, 0.1
G = graph(n_lab, n_train, n_unlab, n_feat, method=method, p=p)
num_epochs, lr = 10, 0.01

#-------------------------------------------------------------#

### MXNET

#from mxnet.gluon import loss as gloss
#import model.mxnet_model as model
#loss = gloss.SoftmaxCrossEntropyLoss()
#model.train(G, loss, num_epochs, lr, method)

#-------------------------------------------------------------#

### Tensorflow

from tensorflow.compat.v1 import losses 
import model.tensorflow_model as model
model.train(G, num_epochs, lr)
