import numpy as np
import tensorflow.compat.v1 as tf

# Apenas para ter
sigm = lambda x: 1. / (1. + tf.math.exp(-x))
deli = lambda a: (a == a.max(axis=1, keepdims=1)).astype(float)

def acc(a, b): return (a * b).sum() / len(a)

#def get_params(G, method):
#    """
#    Get the params for the model.
#    :param G: Grafo de entrada
#    :param method: metodo do modelo
#    :return: retorna lista de parametros do modelo
#    """
#    
#    n_outputs, n_hiddens = 1, 30
# 
#    if method == 1:
#        W1 = tf.Variable(np.random.normal(scale=0.01, size=(1, G.n_feat)))
#        b1 = tf.Variable(tf.zeros((1, 1), dtype=tf.float64))
#        params = [W1, b1]
# 
#    elif method == 2:
#        W1 = tf.Variable(np.random.normal(scale=0.01, size=(n_hiddens, G.n_feat)))
#        b1 = tf.Variable(tf.zeros((n_hiddens, 1), dtype=tf.float64))
#        W2 = tf.Variable(np.random.normal(scale=0.01, size=(n_outputs, n_hiddens)))
#        b2 = tf.Variable(tf.zeros(n_outputs, dtype=tf.float64))
#        params = [W1, b1, W2, b2]
# 
#    return params

# Operações de controle de fluxo
def cond(t1, t2, t3, t4, i, n):
    return tf.less(i, n)

def body(t1, t2, t3, t4, i, n):
    # t1: (l, labels)
    # t2: (k, labels)
    # t3: (k, labels)
    # t4: (k, n)
    return [t1, tf.matmul(t3,tf.matmul(t4, tf.concat([t1, t2], 0))), t3, t4, tf.add(i, 1), n]

#TODO: metodo 1
#    if method == 1:
#        temp = tf.transpose(tf.matmul(theta[0], known) + theta[1])
#        W = tf.reshape(sigm(temp) * adj, shape=(k,n))

def train(G, epochs, lr, method, verbose=True):
    """
    Train and evaluate a model with CPU.
    :param G: Objeto graph com todos os dados
    :param epochs: Número de epochs
    :param lr: Taxa de aprendizado
    :param method: Método do modelo
    :param verbose: Bool para informações da execução na saída padrão
    """

    if(method == 1):
        train1(G, epochs, lr, verbose)
    elif(method == 2):
        train2(G, epochs, lr, verbose)
    else:
        print("Erro: method out of range")

def train1(G, epochs, lr, verbose):

    # Constantes
    labels = G.labels
    
    n = G.vertices
    l = G.n_lab
    k = n-l
    
    n_outputs, n_hiddens = 1, 30
    idx = tf.constant(100)

    # Numeric
    Ytest = G.Ytest.reshape((G.n_unlab, labels))
    Ylabel_numeric = G.Ylabel.reshape((l, labels))
    Ytarget_numeric = G.Ytarget.reshape((G.n_train, labels))
    adj_numeric = G.adj[-k:,].reshape((k*n, 1)).astype(np.float64)
    known_numeric = G.feats[:,-k*n:]

    # Placeholders
    Ylabel = tf.placeholder(dtype=tf.float64, shape=[l, labels])
    Ytarget = tf.placeholder(dtype=tf.float64, shape=[G.n_train, labels])
    adj = tf.placeholder(dtype=tf.float64, shape=[k*n, 1])
    known = tf.placeholder(dtype=tf.float64, shape=[G.n_feat, k*n])

    # Variáveis
    Y = tf.Variable(tf.zeros([G.n_train, labels], dtype=tf.float64))
    W1 = tf.Variable(np.random.normal(scale=0.01, size=(1, G.n_feat)))
    b1 = tf.Variable(tf.zeros((1, 1), dtype=tf.float64))

    W = tf.Variable(tf.zeros((k, n), dtype=tf.float64))
    invA = tf.Variable(tf.zeros((k, k), dtype=tf.float64))

    # Otimizador
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=Ytarget))
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    
    # Inicializa
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Label propagation
    for epoch in range(epochs): 
        prevY = tf.zeros((k, labels), dtype=tf.float64)
        
        ass1 = W.assign(tf.reshape(tf.multiply(tf.transpose(sigm(tf.add(tf.matmul(W1, known), b1))), adj), shape=(k,n)))

        ass2 = invA.assign(tf.transpose(tf.tile(tf.reshape(1/(tf.math.reduce_sum(W, 1))[-k:,], shape=(k,1)), [1, k])))
    
        # Itera até convergir 
        loop = tf.while_loop(cond, body, [Ylabel, prevY, invA, W, 0, idx])
        ass3 = Y.assign(loop[1][:G.n_train])

        # Computa 
        #_ = sess.run(ass1, feed_dict={adj: adj_numeric, known: known_numeric}) 
        #_ = sess.run(ass2)
        #assY = sess.run(ass3, feed_dict={Ylabel: Ylabel_numeric})
        _, l = sess.run([opt, loss], feed_dict={adj: adj_numeric, known: known_numeric, Ylabel: Ylabel_numeric, Ytarget: Ytarget_numeric})

        # Escreve saída
        if verbose:
            print(l/(G.n_train))

            #Yhard = deli(assY)
            #train_l = l / len(assY)
            #train_acc = acc(Yhard[:G.n_train], Ytarget_numeric)
            #test_acc = acc(Yhard[-G.n_unlab:], Ytest)

            #print('epoch {:3d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format( epoch+1, train_l, train_acc, test_acc))

    
    sess.close()

def train2(G, epochs, lr, verbose):

    # Constantes
    labels = G.labels
    
    n = G.vertices
    l = G.n_lab
    k = n-l
    
    n_outputs, n_hiddens = 1, 30
    idx = tf.constant(100)

    # Numeric
    Ytest = G.Ytest.reshape((G.n_unlab, labels))
    Ylabel_numeric = G.Ylabel.reshape((l, labels))
    Ytarget_numeric = G.Ytarget.reshape((G.n_train, labels))
    adj_numeric = G.adj[-k:,].reshape((k*n, 1)).astype(np.float64)
    known_numeric = G.feats[:,-k*n:]

    # Placeholders
    Ylabel = tf.placeholder(dtype=tf.float64, shape=[l, labels])
    Ytarget = tf.placeholder(dtype=tf.float64, shape=[G.n_train, labels])
    adj = tf.placeholder(dtype=tf.float64, shape=[k*n, 1])
    known = tf.placeholder(dtype=tf.float64, shape=[G.n_feat, k*n])

    # Variáveis
    Y = tf.Variable(tf.zeros([G.n_train, labels], dtype=tf.float64))
    W1 = tf.Variable(np.random.normal(scale=0.01, size=(n_hiddens, G.n_feat)))
    b1 = tf.Variable(tf.zeros((n_hiddens, 1), dtype=tf.float64))
    W2 = tf.Variable(np.random.normal(scale=0.01, size=(n_outputs, n_hiddens)))
    b2 = tf.Variable(tf.zeros(n_outputs, dtype=tf.float64))

    W = tf.Variable(tf.zeros((k, n), dtype=tf.float64))
    invA = tf.Variable(tf.zeros((k, k), dtype=tf.float64))

    # Otimizador
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=Ytarget))
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    
    # Inicializa
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Label propagation
    for epoch in range(epochs): 
        prevY = tf.zeros((k, labels), dtype=tf.float64)
        
        ass1 = W.assign(tf.reshape(tf.multiply(tf.transpose(sigm(tf.add(tf.matmul(W2, tf.nn.relu(tf.add(tf.matmul(W1, known), b1))), b2))), adj), shape=(k,n)))

        ass2 = invA.assign(tf.transpose(tf.tile(tf.reshape(1/(tf.math.reduce_sum(W, 1))[-k:,], shape=(k,1)), [1, k])))
    
        # Itera até convergir 
        loop = tf.while_loop(cond, body, [Ylabel, prevY, invA, W, 0, idx])
        ass3 = Y.assign(loop[1][:G.n_train])

        # Computa 
        #_ = sess.run(ass1, feed_dict={adj: adj_numeric, known: known_numeric}) 
        #_ = sess.run(ass2)
        #assY = sess.run(ass3, feed_dict={Ylabel: Ylabel_numeric})
        _, l = sess.run([opt, loss], feed_dict={adj: adj_numeric, known: known_numeric, Ylabel: Ylabel_numeric, Ytarget: Ytarget_numeric})

        # Escreve saída
        if verbose:
            print(l/(G.n_train))

            #Yhard = deli(assY)
            #train_l = l / len(assY)
            #train_acc = acc(Yhard[:G.n_train], Ytarget_numeric)
            #test_acc = acc(Yhard[-G.n_unlab:], Ytest)

            #print('epoch {:3d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format( epoch+1, train_l, train_acc, test_acc))

    
    sess.close()
