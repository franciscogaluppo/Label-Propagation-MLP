# from mxnet import autograd, nd
# from mxnet.gluon import loss as gloss

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, feats, n, V2_to_E, Ylabel):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,1) # this layer does W_1 x + b_1
        self.m = torch.nn.Softmax(dim=1)
        self.feats = feats
        #self.i = torch.LongTensor([row_idx,col_idx])
        #self.n = np.max([row_idx,col_idx])+1
        self.n = n
        self.labels = Ylabel.shape[1]
        self.V2_to_E = V2_to_E
        self.Ylabel = Ylabel

        #self.emb = Variable(torch.randn(self.n,10), requires_grad=True)
        #self.W2  = Variable(torch.randn(10+self.labels,self.labels), requires_grad=True)


    def forward(self, yhat):
        weights = torch.sigmoid(self.fc1(self.feats))

        #pdb.set_trace()
        W = torch.reshape(torch.mm(self.V2_to_E,weights),torch.Size([self.n,self.n]))[self.Ylabel.shape[0]:,:]
        #W = F.normalize(W,p=1,dim=0)

        #emb = torch.cat((self.emb,torch.cat((self.Ylabel,yhat))),dim=1)
        for _ in range(3):
            yhat = torch.mm(W,torch.cat((self.Ylabel,yhat)))
            #emb = torch.mm(W,emb)

        #yhat = torch.mm(emb[self.Ylabel.shape[0]:],self.W2)
        #yhat = self.m(yhat)
        yhat = F.normalize(yhat,p=1,dim=1)

        return yhat




# sigm = lambda x: 1. / (1. + nd.exp(-x))
# def deli(a):
#     (a == a.max(axis=1, keepdims=1)).astype(float)
#     for i in range(len(a)):
#         if(a[i,:].sum() > 1):
#             a[i,:] *= 0
#     return a

# def acc(a, b): return (a * b).sum() / len(a)

# def sgd(theta, lr):
#     """
#     Gradient descent.
#     :param theta: Os parametros do modelo
#     :param lr: Taxa de aprendizado
#     """
#     for param in theta:
#         param[:] = param - lr * param.grad


# def weight_matrix(G, theta, method):
#     """
#     Calcula os pesos das arestas com base nas features
#     :param G: Grafo de entrada
#     :param theta: Os parametros do modelo
#     :param method: O método para cáluclo dos pesos
#     :return: Matriz de pesos das arestas
#     """
#     n = G.vertices
#     l = G.n_lab
#     k = n-l

#     adj = nd.reshape(nd.array(G.adj[-k:,]), shape=(k*n, 1))
#     known = nd.array(G.feats[:,-k*n:])
    
#     # Calcula pesos
#     if method == 1:
#         temp = nd.transpose(nd.dot(theta[0], known) + theta[1])
#         W = nd.reshape(sigm(temp) * adj, shape=(k,n))

#     elif method == 2:
#         temp = nd.relu(nd.dot(theta[0], known) + theta[1])
#         temp = nd.transpose(sigm(nd.dot(theta[2], temp) + theta[3])) * adj
#         W = nd.reshape(temp, shape=(k,n))

#     return W


# def get_params(G, method):
#     """
#     Get the params for the model.
#     :param G: Grafo de entrada
#     :param method: metodo do modelo
#     :return: retorna lista de parametros do modelo
#     """
    
#     n_outputs, n_hiddens = 1, 30

#     if method == 1:
#         W1 = nd.random.normal(scale=0.01, shape=(1, G.n_feat))
#         b1 = nd.zeros((1, 1))
#         params = [W1, b1]

#     elif method == 2:
#         W1 = nd.random.normal(scale=0.01, shape=(n_hiddens, G.n_feat))
#         b1 = nd.zeros((n_hiddens, 1))
#         W2 = nd.random.normal(scale=0.01, shape=(n_outputs, n_hiddens))
#         b2 = nd.zeros(n_outputs)
#         params = [W1, b1, W2, b2]

#     for param in params:
#         param.attach_grad()

#     return params


def train(G, epochs, theta, lr, method=1, verbose=True):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param epochs: numéro de epochs
    :param theta: Parametros do modelo
    :param lr: Taxa de aprendizado
    """

    edges = G.edges # (i->j)
    row_idx, col_idx = zip(*G.edge_list) #i, j
    row_idx = np.array(row_idx) #i
    col_idx = np.array(col_idx) #j
    nz = col_idx*G.vertices+row_idx #j*n + i

    i = torch.LongTensor([[idx, ix] for ix,idx in enumerate(nz)])
    v = torch.FloatTensor([1]*len(nz))
    V2_to_E = torch.sparse.FloatTensor(i.t(), v, torch.Size([G.vertices*G.vertices,len(nz)]) )
    V2_to_E = V2_to_E.coalesce()

    net = Net(torch.tensor(G.feats[:,nz].T).float(), G.vertices, V2_to_E, torch.tensor(G.Ylabel).float())

    # inicializar cada linha uniforme no K-simplex, onde K é o numero de classes
    y_pred = np.random.dirichlet([1]*G.labels,G.n_train+G.n_unlab)
    y_pred = torch.tensor(y_pred).float()
    y_target = torch.tensor(np.argmax(G.Ytarget,axis=1))
    y_test = torch.tensor(np.argmax(G.Ytest,axis=1))

    #pdb.set_trace()
    idx = range(0,y_target.shape[0],round(y_target.shape[0]/10))

    for epoch in range(epochs):
        y_pred = net(y_pred)
        train_loss = F.cross_entropy(y_pred[:G.n_train], y_target)
        test_loss  = F.cross_entropy(y_pred[G.n_train:], y_test)
        print('epoch: {}, train_error: {}, test_error: {}'.format(epoch,train_loss.item(),test_loss.item())
        print(np.argmax(y_pred[idx].detach().numpy(),axis=1),y_target[idx] )
        #pdb.set_trace()
        train_loss.backward(retain_graph=True)

    # adj_numeric = torch.sparse.FloatTensor(G.adj[-k:,].reshape((k*n, 1)).astype(np.float64))

    # loss = gloss.SoftmaxCrossEntropyLoss(sparse_label=False)

    # # Fixos
    # labels = G.labels
    # k = G.vertices - G.n_lab
    # Ytest = G.Ytest.reshape((G.n_unlab, labels))
    # Ytarget_numeric = G.Ytarget.reshape(G.n_train, labels)

    # # Vetores
    # Ylabel = nd.array(G.Ylabel).reshape(G.n_lab, labels)
    # Ytarget = nd.array(Ytarget_numeric)
    # theta = get_params(G, method)
        
    # for epoch in range(epochs): 
    #     prevY = nd.zeros((k, labels))
        
    #     # Calcula erro
    #     with autograd.record():
    #         W = weight_matrix(G, theta, method)
    #         invA = nd.reshape(1/(nd.sum(W, 1))[-k:,], shape=(k,1))

    #         for i in range(100):
    #             concat = nd.concat(Ylabel, prevY, dim=0)
    #             Y = invA * nd.dot(W, concat)

    #             if nd.norm(prevY-Y) < 0.01:
    #                 break
    #             prevY = Y

    #         l = loss(Y[:G.n_train,:], Ytarget).sum()
        
    #     # Atualiza autograds
    #     l.backward()
    #     sgd(theta, lr) 
        
    #     # Escreve saída
    #     if verbose:
    #         Y = Y.asnumpy()
    #         Yhard = deli(Y)
    #         train_l = l.asscalar() / len(Y)
    #         train_acc = acc(Yhard[:G.n_train], Ytarget_numeric)
    #         test_acc = acc(Yhard[-G.n_unlab:], Ytest)

    #         print('epoch {:3d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format(epoch+1, train_l, train_acc, test_acc))
