import numpy as np

class graph():
    """
    Classe de 
    """

    def create_edges(self):
        n = self.vertices

        # Cria matriz sem pesos para o grafo
        self.adj = np.random.randint(0, 2, size=(n,n))
        for i in range(n):
            self.adj[i][i] = 0
        
        # Cria matriz A
        self.D = np.array([sum(adj[i]) for i in range(n)])
        self.A = np.zeros((n,n))
        for i in range(n):
            self.A[(i,i)] = D[i]

        # Guarda as arestas 
        self.edges = sum(self.adj)
        self.edge_list = [(i, j) for i in range(n)
                                 for j in range(n)
                                 if self.adj[(i,j)]]

    def __init__(self, n_unlab=500, n_lab=100, n_train=100, n_feat=100):
        self.vertices = n_unlab + n_lab + n_train

        self.create_edges()

        X = np.random.normal(shape=(self.edges, n_feat))
        self.X_unlab, 
        
