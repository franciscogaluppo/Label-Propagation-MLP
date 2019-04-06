import d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

# 




# Parmetros do modelo
n_feat, n_outputs, n_hiddens = 10, 2, 30

W1 = nd.random.normal(scale=0.01, shape=(n_feat, n_hiddens))
b1 = nd.zeros(n_hiddens)

W2 = nd.random.normal(scale=0.01, shape=(n_hiddens, n_outputs))
b2 = nd.zeros(n_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()



# Modelo
def net(X):
    H = nd.relu(nd.dot(X, W1) + b1)

