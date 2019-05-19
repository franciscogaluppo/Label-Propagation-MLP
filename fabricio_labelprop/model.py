from mxnet import autograd, nd
import numpy as np

def sgd(theta, lr):
    """
    Gradient descent.
    :param theta: Os parametros do modelo
    :param lr: Taxa de aprendizado
    """
    for param in theta:
        param[:] = param - lr * param.grad


def weight_matrix(n, feats, adj, theta):
    """
    Calcula os pesos das arestas com base nas features
    :param n: Número de vértices
    :param feats: As features de cada par de vértices
    :param adj: Matriz de adjacência do grafo
    :param theta: Os parametros do modelo
    """
    
    reLu = lambda x: nd.maximum(x, 0)

    # Calcula pesos
    adj = nd.reshape(nd.array(adj), shape=(n*n))
    # W = nd.sparse.abs(nd.reshape((nd.dot(theta[2],reLu(
    #     nd.dot(theta[0], nd.array(feats)) + theta[1])) + theta[3])*adj, shape=(n,n)))
    W = nd.reshape(nd.sigmoid(
        nd.dot(theta[0], nd.array(feats))+theta[1])*adj, shape=(n,n))

    return W


def train(G, loss, epochs, theta, lr):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param loss: Função de perda
    :param epochs: numéro de epochs
    :param theta: Parametros do modelo
    :param lr: Taxa de aprendizado
    """

    n = len(G.Y0)
    Y0 = nd.array(G.Y0).reshape(n,1)
    Ytrain = nd.array(G.Ytrain).reshape(G.n_train,1)
    Ytest = nd.array(G.Ytest).reshape(len(G.Ytest),1)
    I = nd.concat( nd.ones(G.n_lab), nd.zeros(n-G.n_lab), dim=0 ).reshape((n,1))
    mask_lab = 1-I

    W = nd.array(G.adj)
    Y = nd.dot(W, Y0)
    Y = Y*mask_lab + Y0
    epoch = 0
    train_acc_sum = (
        (Y[G.n_lab:G.n_lab+G.n_train]>0) == (Ytrain>0)).sum().asscalar()
    test_acc_sum = (
        (Y[G.n_lab+G.n_train:]>0) == (Ytest>0)).sum().asscalar()
    print('START: epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:3f}'.format(
        epoch+1, -1, train_acc_sum/G.n_train, test_acc_sum/G.n_unlab ))

    for epoch in range(epochs):         
        prevY = Y0
        converged = False
        
        # Calcula erro
        with autograd.record():
            W = weight_matrix(G.vertices, G.feats, G.adj, theta)
            D = nd.sum_axis(W, 1,keepdims=True)
            #invA = 1./(I+D)
            invA = 1./D            
            for i in range(10):
                Y = invA * (nd.dot(W, prevY))
                Y = Y*mask_lab + Y0
                if nd.norm(prevY-Y) < 0.01:
                    converged = True

                prevY = Y
                if converged:
                    break
            #print(Y)
            #print(Ytrain)
            l = loss(Y[G.n_lab:G.n_lab+G.n_train], Ytrain).sum()
        
        # Atualiza autograds
        l.backward()
        sgd(theta, lr)
        print('W1:',theta[0])
        print('b1:',theta[1])
        W = weight_matrix(G.vertices, G.feats, G.adj, theta)
        D = nd.sum_axis(W, 1,keepdims=True)
        #print(W)
        
        # Escreve saída
        Y = Y.astype('float32')
        train_l_sum = l.asscalar()
        #print('Ytrain:',Ytrain)
        #print('Y:',Y[G.n_lab:G.n_lab+G.n_train])
        # train_acc_sum = (
        #     (Y[G.n_lab:G.n_lab+G.n_train]>0) == (Ytrain>0)).sum().asscalar()
        # test_acc_sum = (
        #     (Y[G.n_lab+G.n_train:]>0) == (Ytest>0)).sum().asscalar()
        train_acc_sum = (
            (-1+2*(Y>0))[G.n_lab:G.n_lab+G.n_train] == Ytrain).sum().asscalar()
        test_acc_sum = (
            (-1+2*(Y>0))[G.n_lab+G.n_train:] == Ytest).sum().asscalar()

        #TODO: test_acc = evaluate_accuracy(test_iter) 
        
        print('epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:3f}'.format(
            epoch+1, train_l_sum/len(Ytrain), train_acc_sum/G.n_train, test_acc_sum/G.n_unlab ))

    print(1./D*W)

