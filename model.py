from mxnet import autograd, nd

def sgd(theta, lr):
    """
    Gradient descent.
    :param theta: Os parametros do modelo
    :param lr: Taxa de aprendizado
    """
    for param in theta:
        param[:] = param - lr * param.grad


def weight_matrix(n, l, feats, adj, theta):
    """
    Calcula os pesos das arestas com base nas features
    :param n: Número de vértices
    :param l: Número de rótulos conhecidos
    :param feats: As features de cada (a, b), rótulo de a desconhecido
    :param adj: Matriz de adjacência do grafo
    :param theta: Os parametros do modelo
    """
    
    k = n-l
    adj = nd.reshape(nd.array(adj[n*l:,]), shape=(k*n))
    
    # Calcula pesos
    temp = nd.relu(nd.dot(theta[0], nd.array(feats[:,(n*l):])) + theta[1])
    temp = nd.reshape((nd.dot(theta[2],temp) + theta[3])*adj,shape=(k,n))
    W = 1/(1 + nd.exp(-temp))

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
        Yl = nd.array(G.Yl).reshape(G.n_lab, 1)
        Ytrain = nd.array(G.Ytrain)
        
        prevY = nd.zeros((G.vertices - G.n_lab, 1))
        converged = False
        
        # Calcula erro
        with autograd.record():
            W = weight_matrix(G.vertices, G.n_lab, G.feats, G.adj, theta)
            D = nd.sum(W, 1)
            invA = nd.diag(1/D)

            while not converged:
                Y = nd.dot(invA, nd.dot(W, nd.concat(Yl, prevY, dim=0)))

                print(D)
                if nd.norm(prevY-Y) < 0.01:
                    converged = True
                prevY = Y

            l = loss(Y, Ytrain).sum()
        
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

