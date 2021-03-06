import numpy as np
import tensorflow.compat.v1 as tf
import numpy as np

# Apenas para ter
def deli(a):
    (a == a.max(axis=1, keepdims=1)).astype(float)
    for i in range(len(a)):
        if(a[i,:].sum() > 1):
            a[i,:] *= 0
    return a

def acc(a, b):
    return (a * b).sum() / len(a)

# Operações de controle de fluxo
def cond(t1, t2, t3, t4, i, n):
    return tf.less(i, n)

def body(t1, t2, t3, t4, i, n):
    # t1: (l, labels)  Ylabel
    # t2: (k, labels)  prevY
    # t3: (k, labels)  invA
    # t4: (k, n)       W

    # TALVEZ MUDE
    return [t1, tf.matmul(t3,tf.matmul(t4, tf.concat([t1, t2], 0))), t3, t4, tf.add(i, 1), n]


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
    
    n = G.vertices
    l = G.n_lab
    k = n-l
    
    n_outputs, n_hiddens = 1, 30
    idx = tf.constant(100)
    prevY = tf.zeros((k, labels), dtype=tf.float64)

    # Numeric
    Ytest = G.Ytest.reshape((G.n_unlab, labels))
    Ylabel_numeric = G.Ylabel.reshape((l, labels))
    Ytarget_numeric = G.Ytarget.reshape((G.n_train, labels))
    #adj_numeric = sparse_diag(G.adj[-k:,].reshape((k*n, 1)).astype(np.float64))
    adj_numeric = G.adj_sparse
    #known_numeric = sparse(G.feats[:,-k*n:])
    known_numeric = G.feats_sparse

    # Placeholders
    Ylabel = tf.placeholder(dtype=tf.float64, shape=[l, labels])
    Ytarget = tf.placeholder(dtype=tf.float64, shape=[G.n_train, labels])
    adj = tf.sparse.placeholder(dtype=tf.float64, shape=[k*n, k*n])
    known = tf.sparse.placeholder(dtype=tf.float64, shape=[G.n_feat, k*n])


    if(method == 1):
        # Variáveis
        W1 = tf.Variable(np.random.normal(scale=0.01, size=(1, G.n_feat)))
        b1 = tf.Variable(tf.zeros((1, 1), dtype=tf.float64))
        
        # Modelo
        tmp = tf.sparse.sparse_dense_matmul(known, W1, adjoint_a=True, adjoint_b=True)
        tmp = tf.transpose(tmp)
        tmp = tf.add(tmp, b1)
        tmp = tf.math.sigmoid(tmp)
        tmp = tf.sparse.sparse_dense_matmul(adj, tmp, adjoint_a=True, adjoint_b=True)
        W = tf.reshape(tmp, shape=(k,n))

    elif(method == 2):
        # Variáveis
        W1 = tf.Variable(np.random.normal(scale=0.01, size=(n_hiddens, G.n_feat)))
        b1 = tf.Variable(tf.zeros((n_hiddens, 1), dtype=tf.float64))
        W2 = tf.Variable(np.random.normal(scale=0.01, size=(n_outputs, n_hiddens)))
        b2 = tf.Variable(tf.zeros(n_outputs, dtype=tf.float64))

        # Modelo
        tmp = tf.sparse.sparse_dense_matmul(known, W1, adjoint_a=True, adjoint_b=True)
        tmp = tf.transpose(tmp)
        tmp = tf.add(tmp, b1)
        tmp = tf.nn.relu(tmp)
        tmp = tf.matmul(W2, tmp)
        tmp = tf.add(tmp, b2)
        tmp = tf.math.sigmoid(tmp)
        tmp = tf.sparse.sparse_dense_matmul(adj, tmp, adjoint_a=True, adjoint_b=True)
        W = tf.reshape(tmp, shape=(k,n))

    else: return

    invA = tf.linalg.tensor_diag(tf.divide(1., (tf.math.reduce_sum(W, 1))[-k:,]))
    Y = tf.while_loop(cond, body, [Ylabel, prevY, invA, W, 0, idx])[1][:G.n_train]

    # Otimizador
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=Ytarget))
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(
        loss, var_list=[W1, b1] if method == 1 else [W1, b1, W2, b2])
    
    # Inicializa
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Label propagation
    for epoch in range(epochs): 
        # Computa 
        _, l, Y_numeric = sess.run([opt, loss, Y],
            feed_dict={adj: adj_numeric,
                       known: known_numeric,
                       Ylabel: Ylabel_numeric,
                       Ytarget: Ytarget_numeric})

        # Escreve saída
        if verbose:
            Yhard = deli(Y_numeric)
            train_l = l / len(Y_numeric)
            train_acc = acc(Yhard[:G.n_train], Ytarget_numeric)
            test_acc = acc(Yhard[-G.n_unlab:], Ytest)

            print('epoch {:3d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format( epoch+1, train_l, train_acc, test_acc))

    sess.close()
