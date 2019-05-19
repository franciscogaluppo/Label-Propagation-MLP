#import d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

import graph
import model
import sys

# Cria um grafo exemplo
# n_lab, n_train, n_unlab, n_feat, = 20, 100, 680, 30
# G = graph.graph(n_lab, n_train, n_unlab, n_feat)
n_lab, n_train, n_unlab, n_feat, = 40, 30, 30, 2
G = graph.graph(n_lab, n_train, n_unlab, n_feat, p=.1)
#G.animate_convergence(save=True)
#sys.exit(0)

# Parametros do modelo
n_outputs, n_hidden = 1, 2

#W1 = nd.random.normal(scale=0.01, shape=(n_hidden, n_feat))
#b1 = nd.zeros((n_hidden, 1))
W1 = nd.random.normal(scale=0.01, shape=(1, n_feat))
b1 = nd.zeros((1, 1))

W2 = nd.random.normal(scale=0.01, shape=(n_outputs, n_hidden))
b2 = nd.zeros(n_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


# Treina o modelo
loss = gloss.L2Loss()
num_epochs, lr = 20, 10
model.train(G, loss, num_epochs, params, lr)
