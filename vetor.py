import numpy as np

n = 5
A = np.array([[0,1,0,0,1],
              [1,0,1,0,0],
              [0,1,0,1,1],
              [0,0,1,0,1],
              [1,0,1,1,0]])

F1 = np.array([[0,0],
               [4,5],
               [0,0],
               [0,0],
               [1,2],
               [4,5],
               [0,0],
               [1,3],
               [0,0],
               [0,0],
               [0,0],
               [1,3],
               [0,0],
               [2,2],
               [3,1],
               [0,0],
               [0,0],
               [2,2],
               [0,0],
               [1,2],
               [1,2],
               [0,0],
               [3,1],
               [1,2],
               [0,0]])

F2 = np.array([[4,5],
               [1,2],
               [1,3],
               [2,2],
               [3,1],
               [1,2]])

vA = np.copy(A)
vA.shape = (n*n,)

edge_list = [(i, j)
    for i in range(n) for j in range(n) if A[(i,j)] and j > i]
print(len(edge_list))

# Parametros do modelo
n_outputs, n_hiddens, n_feat = 1, 30, 2

W1 = np.random.normal(scale=0.01, size=(n_hiddens, n_feat))
b1 = np.random.normal(n_hiddens)

W2 = np.random.normal(scale=0.01, size=(n_outputs, n_hiddens))
b2 = np.random.normal(n_outputs)

theta = [W1, b1, W2, b2]


reLu = lambda x: np.maximum(x, 0, x)


# Método 1
X = np.zeros((n, n))
for e in range(len(edge_list)):
    i, j = edge_list[e]
    X[(i,j)] = theta[2] @ reLu(theta[0] @ np.array(F2[e]) + theta[1]) + theta[3]
    X[(j,i)] = X[(i,j)]
print(X)


# Método 2
Y = theta[2] @ reLu(theta[0] @ np.array(F1).T + theta[1]) + theta[3]
Y *= vA
Y.shape = (n,n)
print(Y)


print(X-Y)
