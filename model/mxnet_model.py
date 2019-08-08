from mxnet import autograd, nd
from mxnet.gluon import loss as gloss

sigm = lambda x: 1. / (1. + nd.exp(-x))
def deli(a):
    (a == a.max(axis=1, keepdims=1)).astype(float)
    for i in range(len(a)):
        if(a[i,:].sum() > 1):
            a[i,:] *= 0
    return a

def acc(a, b): return (a * b).sum() / len(a)

def sgd(theta, lr):
    """
    Gradient descent.
    :param theta: Os parametros do modelo
    :param lr: Taxa de aprendizado
    """
    for param in theta:
        param[:] = param - lr * param.grad


def weight_matrix(G, theta, method):
    """
    Calcula os pesos das arestas com base nas features
    :param G: Grafo de entrada
    :param theta: Os parametros do modelo
    :param method: O método para cáluclo dos pesos
    :return: Matriz de pesos das arestas
    """
    n = G.vertices
    l = G.n_lab
    k = n-l

    adj = nd.reshape(nd.array(G.adj[-k:,]), shape=(k*n, 1))
    known = nd.array(G.feats[:,-k*n:])
    
    # Calcula pesos
    if method == 1:
        temp = nd.transpose(nd.dot(theta[0], known) + theta[1])
        W = nd.reshape(sigm(temp) * adj, shape=(k,n))

    elif method == 2:
        temp = nd.relu(nd.dot(theta[0], known) + theta[1])
        temp = nd.transpose(sigm(nd.dot(theta[2], temp) + theta[3])) * adj
        W = nd.reshape(temp, shape=(k,n))

    return W


def get_params(G, method):
    """
    Get the params for the model.
    :param G: Grafo de entrada
    :param method: metodo do modelo
    :return: retorna lista de parametros do modelo
    """
    
    n_outputs, n_hiddens = 1, 30

    if method == 1:
        W1 = nd.random.normal(scale=0.01, shape=(1, G.n_feat))
        b1 = nd.zeros((1, 1))
        params = [W1, b1]

    elif method == 2:
        W1 = nd.random.normal(scale=0.01, shape=(n_hiddens, G.n_feat))
        b1 = nd.zeros((n_hiddens, 1))
        W2 = nd.random.normal(scale=0.01, shape=(n_outputs, n_hiddens))
        b2 = nd.zeros(n_outputs)
        params = [W1, b1, W2, b2]

    for param in params:
        param.attach_grad()

    return params


def train(G, epochs, theta, lr, method=1, verbose=True):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param epochs: numéro de epochs
    :param theta: Parametros do modelo
    :param lr: Taxa de aprendizado
    """

    loss = gloss.SoftmaxCrossEntropyLoss(sparse_label=False)

    # Fixos
    labels = G.labels
    k = G.vertices - G.n_lab
    Ytest = G.Ytest.reshape((G.n_unlab, labels))
    Ytarget_numeric = G.Ytarget.reshape(G.n_train, labels)

    # Vetores
    Ylabel = nd.array(G.Ylabel).reshape(G.n_lab, labels)
    Ytarget = nd.array(Ytarget_numeric)
    theta = get_params(G, method)
        
    for epoch in range(epochs): 
        prevY = nd.zeros((k, labels))
        
        # Calcula erro
        with autograd.record():
            W = weight_matrix(G, theta, method)
            invA = nd.reshape(1/(nd.sum(W, 1))[-k:,], shape=(k,1))

            for i in range(100):
                concat = nd.concat(Ylabel, prevY, dim=0)
                Y = invA * nd.dot(W, concat)

                if nd.norm(prevY-Y) < 0.01:
                    break
                prevY = Y

            l = loss(Y[:G.n_train,:], Ytarget).sum()
        
        # Atualiza autograds
        l.backward()
        sgd(theta, lr) 
        
        # Escreve saída
        if verbose:
            Y = Y.asnumpy()
            Yhard = deli(Y)
            train_l = l.asscalar() / len(Y)
            train_acc = acc(Yhard[:G.n_train], Ytarget_numeric)
            test_acc = acc(Yhard[-G.n_unlab:], Ytest)

            print('epoch {:3d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format(epoch+1, train_l, train_acc, test_acc))
