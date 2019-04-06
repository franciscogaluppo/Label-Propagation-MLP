import numpy as np
from copy import copy

# Label propagation do link que o murai mandou
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
