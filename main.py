#import d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

import graph
import model

# Cria um grafo exemplo
method = 2
n_lab, n_train, n_unlab, n_feat = 20, 100, 680, 30
G = graph.random_graph(n_lab, n_train, n_unlab, n_feat, method=method)

# Treina o modelo
loss = gloss.L2Loss()
num_epochs, lr = 10, 0.01
model.train(G, loss, num_epochs, lr, method)
