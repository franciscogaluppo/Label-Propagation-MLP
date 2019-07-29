from graphs.random_features import random_features as graph

# Cria um grafo exemplo
method, labels = 2, 2
n_lab, n_train, n_unlab, n_feat, p = 20, 60, 20, 10, 0.3
G = graph(n_lab, n_train, n_unlab, n_feat, labels, method, p=p)
num_epochs, lr = 10, 0.01

#-------------------------------------------------------------#

### MXNET

print("MXNET")
import model.mxnet_model as model1
model1.train(G, num_epochs, lr, method)

#-------------------------------------------------------------#

### Tensorflow

print("\n\nTENSORFLOW")
import model.tensorflow_model as model2
model2.train(G, num_epochs, lr, method)
