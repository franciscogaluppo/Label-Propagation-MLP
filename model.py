from mxnet import autograd, nd

def sgd(theta, lr):
    """
    Stochastic gradient descent.
    :param theta: Os parametros do modelo
    :param lr: Taxa de aprendizado
    """
    for param in theta:
        param[:] = param - lr * param.grad


# TODO: parmetros
def weight_matrix(n, feats, adj, theta):
    """
    Calcula os pesos das arestas com base nas features
    :param n:
    :param feats: As features de cada par de vértices
    :param adj: 
    :param theta: Os parametros do modelo
    """
    
    reLu = lambda x: nd.maximum(x, 0)

    # Calcula pesos
    adj = nd.reshape(nd.array(adj), shape=(n*n))
    W = nd.sparse.abs(nd.reshape((nd.dot(theta[2],reLu(
        nd.dot(theta[0], nd.array(feats)) + theta[1])) + theta[3])*adj, shape=(n,n)))

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
    
    for epoch in range(epochs): 
        Y0 = nd.array(G.Y0)
        Ytrain = nd.array(G.Ytrain)
        I = nd.concat(nd.ones(G.n_lab), nd.zeros(len(Y0) - G.n_lab),dim=0)
        
        prevY = Y0
        converged = False
        
        # Calcula erro
        with autograd.record():
            W = weight_matrix(G.vertices, G.feats, G.adj, theta)
            D = nd.sum(W, 0)
            invA = nd.diag(1/(I+D))
            
            while not converged:
                Y = nd.dot(invA, (nd.dot(W, prevY) + Y0))

                if nd.norm(prevY-Y) < 0.01:
                    converged = True
                prevY = Y
            l = loss(Y[G.n_lab:], Ytrain).sum()
        
        # Atualiza autograds
        l.backward()
        sgd(theta, lr) 
        
        # Escreve saída
        Y = Y.astype('float32')
        train_l_sum = l.asscalar()
        train_acc_sum = (
            (-1+2*(Y>0))[G.n_lab:][:G.n_train] == nd.array(G.Ytrain[:G.n_train])).sum().asscalar()

        #TODO: test_acc = evaluate_accuracy(test_iter) 
        
        print('epoch {}, loss {:.4f}, train acc {:.3f}'.format(
            epoch+1, train_l_sum/len(Ytrain), train_acc_sum/(G.n_train)))

