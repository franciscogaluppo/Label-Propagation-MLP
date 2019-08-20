import numpy as np
import tensorflow as tf
from scipy import sparse

a = np.array([[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0.2, 0.4, 0],
             [0, 0, 0],
             [0, 1, 4],
             [0, np.pi, 0]])

sp = sparse.csr_matrix(a)
coo = sp.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
tsp = tf.SparseTensor(indices, coo.data, coo.shape)

with tf.Session() as s:
    print(s.run(tsp))
