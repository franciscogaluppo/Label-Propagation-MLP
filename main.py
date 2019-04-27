#import d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

import graph
import model

# Cria um grafo exemplo
n_lab, n_train, n_unlab, n_feat = 10, 50, 340, 30
G = graph.graph(n_lab, n_train, n_unlab, n_feat)


# Parametros do modelo
n_outputs, n_hiddens = 1, 30

W1 = nd.random.normal(scale=0.01, shape=(n_hiddens, n_feat))
b1 = nd.zeros(n_hiddens)

W2 = nd.random.normal(scale=0.01, shape=(n_outputs, n_hiddens))
b2 = nd.zeros(n_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


# Treina o modelo
loss = gloss.L2Loss()
num_epochs, lr = 10, 0.5
model.train(G, loss, num_epochs, params, lr)
