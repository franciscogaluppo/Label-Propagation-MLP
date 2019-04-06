import numpy as np

# TODO: Não podemos ter vértices isolados. A não teria inversa
# TODO: Esqueci que tem que ser direcionado...

class graph():
    """
    Classe que gera grafos
    """

    def create_edges(self):
        n = self.vertices

        # Cria matriz de adjacência
        self.adj = np.random.randint(0, 2, size=(n,n))
        for i in range(n):
            self.adj[i][i] = 0

        # Cria matriz A
        self.D = np.array([sum(self.adj[i]) for i in range(n)])
        self.A = np.zeros((n,n))
        for i in range(n):
            self.A[(i,i)] = self.D[i]


        # Guarda as arestas 
        self.edges = np.sum(self.adj)
        self.edge_list = [(i, j) for i in range(n)
                                 for j in range(n)
                                 if self.adj[(i,j)]]

    def __init__(self, n_unlab=500, n_lab=100, n_train=100, n_feat=100):
        
        # Guarda os valores
        self.vertices = n_unlab + n_lab + n_train
        self.n_lab = n_lab
        self.n_unlab = n_unlab
        self.n_train = n_train

        # Cria matrix e atualiza a matriz A
        self.create_edges()
        for i in range(n_lab):
            self.A[(i,i)] += 1
        
        # Inverte a matriz 
        self.inv_A = np.linalg.inv(self.A)

        # Cria as features das arestas
        self.X = np.random.normal(size=(self.edges, n_feat))

        # Cria os vetores de labels
        self.y_unlab = np.zeros(n_unlab)
        self.y_lab = np.ones(n_lab) + np.random.randint(0, 2,size=(n_lab))
        self.y_train=np.ones(n_train)+np.random.randint(0,2,size=(n_train))
        self.y = np.concatenate((self.y_lab, self.y_train, self.y_unlab))


    
