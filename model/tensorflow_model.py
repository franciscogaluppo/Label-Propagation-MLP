import numpy as np
import tensorflow as tf

# Apenas para ter
sigm = lambda x: 1. / (1. + tf.math.exp(-x))
deli = lambda x: 1*(x>0) - 1*(x<0)

def acc(a, b): return (a == b).sum().asscalar() / len(a)

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

    adj = tf.convert_to_tensor(G.adj[-k:,].reshape((k*n, 1)), dtype=tf.float64)
    known = tf.convert_to_tensor(G.feats[:,-k*n:])
    
    # Calcula pesos
    if method == 1:
        temp = tf.transpose(tf.matmul(theta[0], known) + theta[1])
        W = tf.reshape(sigm(temp) * adj, shape=(k,n))

    elif method == 2:
        temp = tf.nn.relu(tf.matmul(theta[0], known) + theta[1])
        temp = tf.transpose(sigm(tf.matmul(theta[2], temp) + theta[3])) * adj
        W = tf.reshape(temp, shape=(k,n))

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
        W1 = tf.convert_to_tensor(np.random.normal(scale=0.01, size=(1, G.n_feat)))
        b1 = tf.zeros((1, 1), dtype=tf.float64)
        params = [W1, b1]

    elif method == 2:
        W1 = tf.convert_to_tensor(np.random.normal(scale=0.01, size=(n_hiddens, G.n_feat)))
        b1 = tf.zeros((n_hiddens, 1))
        W2 = tf.convert_to_tensor(np.random.normal(scale=0.01, size=(n_outputs, n_hiddens)))
        b2 = tf.zeros(n_outputs)
        params = [W1, b1, W2, b2]

    return params


def train(G, loss, epochs, theta, lr, method=1, verbose=True):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param loss: Função de perda
    :param epochs: numéro de epochs
    :param theta: Parametros do modelo
    :param lr: Taxa de aprendizado
    """
    # Vetores fixos
    labels = G.labels
    k = G.vertices - G.n_lab
    Ylabel = tf.reshape(tf.convert_to_tensor(G.Ylabel), shape=(G.n_lab, labels))
    Ytarget = tf.reshape(tf.convert_to_tensor(G.Ytarget), shape=(G.n_train, labels))
    Ytest = tf.reshape(tf.convert_to_tensor(G.Ytest), shape=(G.n_unlab, labels))

    # Parametros e otimizador
    theta = get_params(G, method)
    opt = tf.compat.v1.train.GradientDescentOptimizer(lr)
        
    for epoch in range(epochs): 
        prevY = tf.zeros((G.vertices - G.n_lab, 1))
        
        # Calcula erro
        with tf.GradientTape() as t:
            t.watch(theta)

            W = weight_matrix(G, theta, method)
            invA = tf.reshape(1/(tf.math.reduce_sum(W, 1))[-k:,], shape=(k,1))

            for i in range(100):
                concat = tf.concat(Ylabel, prevY)
                Y = invA * tf.matmul(W, concat)
                if tf.norm(prevY-Y) < 0.01:
                    break
                prevY = Y
        
        # TODO: Loss function
        train = opt.minimize(tf.subtract(Y[:G.n_train,:], Ytarget))

        # Escreve saída
        if verbose:
            Y = Y.astype('float32')
            Yhard = deli(Y)
            train_l = l.asscalar() / len(Y)
            train_acc = acc(Yhard[:G.n_train], Ytarget)
            test_acc = acc(Yhard[-G.n_unlab:], Ytest)

            print('epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:.3}'.format(epoch+1, train_l, train_acc, test_acc))
