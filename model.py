from mxnet import autograd, nd

sigm = lambda x: 1. / (1. + nd.exp(-x))
deli = lambda x: 1*(x>0) - 1*(x<0)

def sgd(theta, lr):
    """
    Gradient descent.
    :param theta: Os parametros do modelo
    :param lr: Taxa de aprendizado
    """
    for param in theta:
        param[:] = param - lr * param.grad


def weight_matrix(n, l, feats, adj, theta, method=1):
    """
    Calcula os pesos das arestas com base nas features
    :param n: Número de vértices
    :param l: Número de rótulos conhecidos
    :param feats: As features de cada (a, b), rótulo de a desconhecido
    :param adj: Matriz de adjacência do grafo
    :param theta: Os parametros do modelo
    :param method: O método para cáluclo dos pesos
    """
    
    k = n-l
    adj = nd.reshape(nd.array(adj[-k:,]), shape=(k*n, 1))
    known = nd.array(feats[:,-k*n:])
    
    # Calcula pesos
    if method == 1:
        temp = nd.relu(nd.dot(theta[0], known) + theta[1])
        temp = nd.transpose(sigm(nd.dot(theta[2], temp) + theta[3])) * adj
        W = nd.reshape(temp, shape=(k,n))

    # TODO: arrumar as dimensões
    elif method == 2:
        temp = nd.dot(theta[0], known) + theta[1] 
        W = nd.reshape(sigm(temp) * adj, shape=(k,n))

    return W


def train(G, loss, epochs, theta, lr, verbose=True):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param loss: Função de perda
    :param epochs: numéro de epochs
    :param theta: Parametros do modelo
    :param lr: Taxa de aprendizado
    """
    # Vetores fixos 
    k = G.vertices - G.n_lab
    Ylabel = nd.array(G.Ylabel).reshape(G.n_lab, 1)
    Ytarget = nd.array(G.Ytarget).reshape(k, 1)
    Ytest = nd.array(G.Ytest).reshape(G.n_unlab, 1)
        
    for epoch in range(epochs): 
        prevY = nd.zeros((G.vertices - G.n_lab, 1))
        
        # Calcula erro
        with autograd.record():
            W = weight_matrix(G.vertices, G.n_lab, G.feats, G.adj, theta)
            invA = nd.reshape(1/(nd.sum(W, 1))[-k:,], shape=(k,1))

            for i in range(100):
                concat = nd.concat(Ylabel, prevY, dim=0)
                Y = invA * nd.dot(W, concat)
                if nd.norm(prevY-Y) < 0.01:
                    break
                prevY = Y

            l = loss(Y, Ytarget).sum()
        
        # Atualiza autograds
        l.backward()
        sgd(theta, lr) 
        
        # Escreve saída
        if verbose:
            Y = Y.astype('float32')
            Yhard = deli(Y)

            train_l = l.asscalar() / len(Y)
            print(Yhard[0], Ytarget[0])
            train_acc = (Yhard[G.n_train:] == Ytarget[G.n_train:])
            print(train_acc[0])
            train_acc = train_acc.sum().asscalar() /  G.n_train

            test_acc = Yhard[-G.n_unlab:] == Ytest
            test_acc = test_acc.sum().asscalar() / G.n_unlab

            print('epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:.3}'.format(
            epoch+1, train_l, train_acc, test_acc))

