from graphs.random_features import random_features as graph

#from mxnet.gluon import loss as gloss
from tensorflow.compat.v1 import losses 

#import model.mxnet_model as model
import model.tensorflow_model as model

# Cria um grafo exemplo
method = 2
n_lab, n_train, n_unlab, n_feat, p = 200, 1000, 100, 30, 0.1
G = graph(n_lab, n_train, n_unlab, n_feat, method=method, p=p)

# Treina o modelo
#loss = gloss.SoftmaxCrossEntropyLoss()
#loss = losses.softmax_cross_entropy(n_train, 1)
loss = 3
num_epochs, lr = 10, 0.01
model.train(G, loss, num_epochs, lr, method)
