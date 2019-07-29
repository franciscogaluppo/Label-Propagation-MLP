import numpy as np
import tensorflow.compat.v1 as tf

# Apenas para ter
sigm = lambda x: 1. / (1. + tf.math.exp(-x))
deli = lambda a: (a == a.max(axis=1, keepdims=1)).astype(float)

def acc(a, b): return (a * b).sum() / len(a)


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

    adj = tf.convert_to_tensor(G.adj[-k:,].reshape((k*n, 1)).astype(np.float64))
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


def get_params(G, method):
    """
    Get the params for the model.
    :param G: Grafo de entrada
    :param method: metodo do modelo
    :return: retorna lista de parametros do modelo
    """
    
    n_outputs, n_hiddens = 1, 30

    if method == 1:
        W1 = tf.Variable(np.random.normal(scale=0.01, size=(1, G.n_feat)))
        b1 = tf.Variable(tf.zeros((1, 1), dtype=tf.float64))
        params = [W1, b1]

    elif method == 2:
        W1 = tf.Variable(np.random.normal(scale=0.01, size=(n_hiddens, G.n_feat)))
        b1 = tf.Variable(tf.zeros((n_hiddens, 1), dtype=tf.float64))
        W2 = tf.Variable(np.random.normal(scale=0.01, size=(n_outputs, n_hiddens)))
        b2 = tf.Variable(tf.zeros(n_outputs, dtype=tf.float64))
        params = [W1, b1, W2, b2]

    return params

# Operações de controle de fluxo
def cond(t1, t2, t3, t4, i, n):
    return tf.less(i, n)

def body(t1, t2, t3, t4, i, n):
    return [t1, t3*tf.matmul(t4, tf.concat([t1, t2], 0)), t3, t4, tf.add(i, 1), n]

def train(G, epochs, lr, method, verbose=True):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param epochs: Número de epochs
    :param lr: Taxa de aprendizado
    :param method: Método do modelo
    :param verbose: Bool para informações da execução na saída padrão
    """

    # Constantes
    labels = G.labels
    k = G.vertices - G.n_lab 
    Ytest = G.Ytest.reshape((G.n_unlab, labels))
    Ylabel_numeric = G.Ylabel.reshape((G.n_lab, labels))
    Ytarget_numeric = G.Ytarget.reshape((G.n_train, labels))
    n = tf.constant(100)

    # Placeholders e variáveis
    Ylabel = tf.placeholder(dtype=tf.float64, shape=[G.n_lab, labels])
    Ytarget = tf.placeholder(dtype=tf.float64, shape=[G.n_train, labels])
    Y = tf.Variable(tf.zeros([G.n_train, labels], dtype=tf.float64))
    theta = get_params(G, method)

    # Otimizador
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=Ytarget))
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    
    # Inicializa
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Label propagation
    for epoch in range(epochs): 
        prevY = tf.zeros((G.vertices - G.n_lab, labels), dtype=tf.float64)
        W = weight_matrix(G, theta, method)
        invA = tf.reshape(1/(tf.math.reduce_sum(W, 1))[-k:,], shape=(k,1))
    
        # Itera até convergir 
        res = tf.while_loop(cond, body, [Ylabel, prevY, invA, W, 0, n])
        ass = Y.assign(res[1][:G.n_train])

        # Computa 
        assY, _, l = sess.run([ass, opt, loss],
            feed_dict={Ylabel: Ylabel_numeric, Ytarget: Ytarget_numeric})

        # Escreve saída
        if verbose:
            Yhard = deli(assY)
            train_l = l / len(assY)
            train_acc = acc(Yhard[:G.n_train], Ytarget_numeric)
            test_acc = acc(Yhard[-G.n_unlab:], Ytest)

            print('epoch {:3d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format( epoch+1, train_l, train_acc, test_acc))

    
    sess.close()
