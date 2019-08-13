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
find = sparse.find(sp)
indices = [x.tolist() for x in find[0:2]]
print(indices)
values = find[2]
shape = a.shape
print(shape)

tsp = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

with tf.Session() as s:
    print(s.run(tsp))
