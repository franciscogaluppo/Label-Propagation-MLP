import numpy as np

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
        
        # Cria as features das arestas
        self.X = np.random.normal(size=(self.edges, self.n_feat))
        
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
        self.W = np.zeros((self.vertices, self.vertices))
        for e in range(self.edges):
            i, j = self.edge_list[e]
            self.W[(i,j)] = W2 @ reLu(W1 @ self.X[e] + b1) + b2
            self.W[(j,i)] = self.W[(i,j)]



