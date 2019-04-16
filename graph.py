import numpy as np
from copy import copy

class graph():
    """
    Classe que gera grafos.
    """

    def __init__(self, n_lab=100, n_train=100, n_unlab=500, n_feat=100):
        """
        Inicializa a classe
        :param n_lab: número de vértices com rótulo conhecido
        :param n_train: número de vértices de treino
        :param n_unlab: número de vértices sem rótulo
        :param n_feat: número de features para cada aresta
        """ 

        # Guarda os valores
        self.vertices = n_lab + n_train + n_unlab
        self.n_lab = n_lab
        self.n_train = n_train
        self.n_unlab = n_unlab
        self.n_feat = n_feat

        # Cria matrix e atualiza a matriz A
        self.create_edges()
        for i in range(n_lab):
            self.A[(i,i)] += 1
        
        # Cria as features das arestas
        self.X = np.random.normal(size=(self.edges, self.n_feat))

        # Faz a chamada da criação dos rótulos
        self.create_labels()

    def create_edges(self):
        """
        Cria a matriz de adjacência, matriz A e lista de arestas.
        """
        n = self.vertices
        
        # Cria matriz de adjacência
        v = np.random.randint(0, 2, n*(n-1)//2)
        self.adj = np.zeros((n,n), dtype=int)
        self.adj[np.tril_indices(self.adj.shape[0],-1)] = v
        self.adj = np.transpose(self.adj)
        self.adj[np.tril_indices(self.adj.shape[0],-1)] = v

        # Checa se existe algum vértice isolado
        for i in range(n):
            if np.sum(self.adj[i]) == 0:

                # Cria uma aresta aleatória nova
                j = i
                while j == i:
                    j = np.random.randint(0, n)
                
                self.adj[(i, j)] = 1
                self.adj[(j, i)] = 1
                
        ##### PRECISA DA MATRIZ DE ADJACÊNCIA????

        # Cria matriz A
        self.D = np.array([sum(self.adj[i]) for i in range(n)])
        self.A = np.zeros((n,n), dtype=int)
        for i in range(n):
            self.A[(i,i)] = self.D[i]

        # Inverte a matriz 
        self.invA = np.linalg.inv(self.A)

        # Guarda as arestas 
        self.edges = int (np.sum(self.adj) / 2)
        self.edge_list = [(i, j) for i in range(n)
                                 for j in range(n)
                                 if self.adj[(i,j)] and j > i]


    def create_labels(self):
        """
        Cria os rótulos para n_lab + n_train vértices.
        Designa alguns rótulos aleatoriamente e propaga.
        Ignora os rótulos dos n_unlab ao final.
        """
        
        n = self.vertices

        # Número oculto
        n_hidden = 30

        # Matrizes ocultas
        W1 = np.random.normal(size=(n_hidden, self.n_feat))
        W2 = np.random.normal(size=n_hidden)

        b1 = np.random.normal(size=n_hidden)
        b2 = np.random.normal()

        # Rectifier
        reLu = lambda x: np.maximum(x, 0, x)

        # Cria os pesos
        W = np.zeros((n, n))
        for e in range(self.edges):
            i, j = self.edge_list[e]
            W[(i,j)] = W2 @ reLu(W1 @ self.X[e] + b1) + (b2 if b2 > 0 else 0)
            W[(j,i)] = W[(i,j)]

        # Gera label inicial para alguns
        n_known = int(n / 10) + 2
        known = (-1)**np.random.randint(0,2,n_known)
        y0 = np.concatenate(((known), np.zeros(n-n_known)))
        y_hat = copy(y0)

        # Matriz A
        D = np.sum(W,0) 
        A = np.diag(np.concatenate((np.ones(n_known), np.zeros(n-n_known))) + D)
     
        # Itera para convergir
        for iter in range(100):
            y_hat_old = y_hat
            y_hat = np.linalg.solve(A, np.dot(W, y_hat) + y0)
     
            if np.linalg.norm(y_hat - y_hat_old) < 0.01:
                break

        # Guarda Rótulos
        self.Y0 = np.concatenate((y_hat[0:(self.n_lab + self.n_train -1)], np.zeros(self.n_unlab)))

        self.Y0_bin = [-1 if x < 0 else (1 if x > 0 else 0) for x in self.Y0]
