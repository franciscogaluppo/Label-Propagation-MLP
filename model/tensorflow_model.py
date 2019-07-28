import numpy as np
import tensorflow.compat.v1 as tf

# Apenas para ter
sigm = lambda x: 1. / (1. + tf.math.exp(-x))
deli = lambda x: 1*(x>0) - 1*(x<0)

def acc(a, b): return (a == b).sum().asscalar() / len(a)

# TODO: SOLVE THIS
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

    adj = tf.convert_to_tensor(G.adj[-k:,].reshape((k*n, 1)))
    known = tf.convert_to_tensor(G.feats[:,-k*n:])
    
    # Calcula pesos
    if method == 1:
        temp = tf.transpose(tf.matmul(theta[0], known) + theta[1])
        W = tf.reshape(sigm(temp) * adj, shape=(k,n))

    elif method == 2:
        temp1 = tf.nn.relu(tf.matmul(theta[0], known) + theta[1])
        temp2 = tf.transpose(sigm(tf.matmul(theta[2], temp1) + theta[3]))
        temp3 = tf.multiply(temp2, adj)
        W = tf.reshape(temp3, shape=(k,n))

    return W


def train(G, epochs, lr, verbose=True):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param epochs: Número de epochs
    :param lr: Taxa de aprendizado
    :param verbose: Bool para informações da execução na saída padrão
    """

    # Constantes
    labels = G.labels
    k = G.vertices - G.n_lab 
    Ytest = G.Ytest.reshape((G.n_unlab, labels))
    Ylabel_numeric = G.Ylabel.reshape((G.n_lab, labels))
    Ytarget_numeric = G.Ytarget.reshape((G.n_train, labels))

    # Placeholders
    Ylabel = tf.placeholder(dtype=tf.float64, shape=[None, G.n_lab, labels])
    Ytarget = tf.placeholder(dtype=tf.float64, shape=[None, G.n_train, labels])
    Y = tf.Variable(tf.zeros([G.n_train, labels], dtype=tf.float64))

    # Otimizador
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=Ytarget))
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    
    # Inicializa
    tf.set_random_seed(1234)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Label propagation
    for epoch in range(epochs): 
        prevY = tf.zeros((G.vertices - G.n_lab, 1))
        
        W = weight_matrix(G, theta, method)
        invA = tf.reshape(1/(tf.math.reduce_sum(W, 1))[-k:,], shape=(k,1))

        for i in range(100):
            concat = tf.concat(Ylabel, prevY)
            currY = invA * tf.matmul(W, concat)
            if tf.norm(prevY-Y) < 0.01:
                break
            prevY = currY
       
        ass = Y.assign(currY[:G.n_train])

        # Computa 
        sess.run(ass)
        sess.run(opt, feed_dict={Ylabel: Ylabel_numeric, Ytarget: Ytarget_numeric})

        # Escreve saída
        if verbose:
            Y = Y.astype('float32')
            Yhard = deli(Y)
            train_l = loss.asscalar() / len(Y)
            train_acc = acc(Yhard[:G.n_train], Ytarget)
            test_acc = acc(Yhard[-G.n_unlab:], Ytest)

            print('epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:.3}'.format(
                epoch+1, train_l, train_acc, test_acc))

    
    sess.close()
