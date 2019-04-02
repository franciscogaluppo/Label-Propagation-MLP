import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from copy import copy

def loss(y, y_hat):
    """
    Função de erro
    :param y: targets do treino
    :param y_hat: estimações do treino
    """
    return(((y - y_hat)**2).sum())

def learn(y_labeled, X_labeled, X_unlabeled, y_train, X_train, mu=1., verbose=False):
    """
    Estima as labels, estimando os pesos com um MLP
    :param y_labeled: targets for labeled points
    :param X_labeled:  data for labeled points
    :param X_unlabeled: data for unlabeled points
    :param X_train: data for where to evaluate the label
    :param kernel: kernel function
    :param mu: hyperparameter for the label prop algo
    :param verbose: do you want to print stuff?
    :return:
    """
    
    for iter in range(100):

        # TODO: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        clf = MLPClassifier()
        clf.fit(X, y)

        #w = ...
        y_hat = label_propagation(y_labeled, X_labeled, X_unlabeled, X_train, W, mu)

        if loss(y_train, y_hat) < 0.01:
            if verbose:
                print('Converged after %i steps'%iter)
            break

        else:
            if verbose:
            print('Not converged??')

    return(y_hat)

def label_propagation(y_labeled, X_labeled, X_unlabeled, X_train, W, mu=1.):
    """
    Label propagation algorithm on the data
    :param y_labeled: targets for labeled points
    :param X_labeled:  data for labeled points
    :param X_unlabeled: data for unlabeled points
    :param X_train: data for where to evaluate the label
    :param kernel: kernel function
    :param mu: hyperparameter for the label prop algo
    :param verbose: do you want to print stuff?
    :return:
    """
    n_unlabeled = X_unlabeled.shape[0]
    n_labeled = X_labeled.shape[0]
    n_train = X_train.shape[0]

    # from here on, we define X_eval as the data points where we want to EVALuate the labels
    X_eval = np.concatenate((X_labeled, X_unlabeled, X_train), 0)

    W = kernel(X_eval, None)
    D = np.sum(W,0)

    eps = 1E-9 #arbitrary small number

    A = np.diag(np.concatenate((np.ones((n_labeled)), np.zeros((n_unlabeled+n_train)))) + mu*D + mu*eps)

    y_hat_0 = np.concatenate((y_labeled, np.zeros((n_unlabeled+n_train))))
    y_hat = copy(y_hat_0)

    for iter in range(100):
        y_hat_old = y_hat
        y_hat = np.linalg.solve(A, mu*np.dot(W, y_hat)+y_hat_0)

        if np.linalg.norm(y_hat - y_hat_old) < 0.01:
            break

return y_hat[-n_train:]
