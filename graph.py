import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

def delimiter(x):
    return list(map(lambda y: 1*(y>0)-1*(y<0), x))



class graph():

    def __init__(self, n_lab=10, n_train=10, n_unlab=50, n_feat=10, p=.1):
        """
        Inicializa a classe
        :param n_lab: número de vértices com rótulo conhecido
        :param n_train: número de vértices de treino
        :param n_unlab: número de vértices sem rótulo
        :param n_feat: número de features para cada aresta
        """ 

        # Guarda os valores
        self.vertices = n_lab + n_train + n_unlab
        self.n_lab, self.n_train = n_lab, n_train
        self.n_unlab, self.n_feat = n_unlab, n_feat

        # Cria grafo e rótulos
        self.create_edges(p)
        self.create_features()
        self.create_labels()


    def create_edges(self, p):
        """
        Cria a matriz de adjacência, matriz A e lista de arestas.
        """
        n = self.vertices
        
        # Cria matriz de adjacência
        v = np.random.binomial(1, p, n*(n-1)//2)
        self.adj = np.zeros((n,n), dtype=int)
        self.adj[np.tril_indices(self.adj.shape[0],-1)] = v
        self.adj = np.transpose(self.adj)
        self.adj[np.tril_indices(self.adj.shape[0],-1)] = v

        # "Conserta" os vértices isolados
        for i in range(n):
            if np.sum(self.adj[i]) == 0:

                j = np.random.randint(0, n)
                while j == i:
                    j = np.random.randint(0, n)
                
                self.adj[(i, j)] = self.adj[(j, i)] = 1
                
        # Guarda as arestas 
        self.edges = int (np.sum(self.adj) / 2)
        self.edge_list = [(i, j) for i in range(n)
                                 for j in range(n)
                                 if self.adj[(i,j)] and j > i]


    def create_features(self):
        """
        Cria as features para todo par de vértices.
        A feature é de zeros quando não existe tal aresta.
        """
        
        n = self.vertices

        # Cria features e as coloca no formato padrão
        X = np.random.normal(size=(self.edges, self.n_feat))
        feats = np.zeros((n*n, self.n_feat))
        for e in range(self.edges):
            i, j = self.edge_list[e]
            feats[i*n+j] = feats[j*n+i] = X[e]
        self.feats = feats.T



    def create_labels(self):
        """
        Cria os rótulos para n_lab + n_train vértices.
        Designa alguns rótulos aleatoriamente e propaga.
        Ignora os rótulos dos n_unlab ao final.
        """
        
        n = self.vertices

        # Rectifier
        reLu = lambda x: np.maximum(x, 0, x)

        # Número oculto
        n_hidden = 30

        # Matrizes ocultas
        W1 = np.random.normal(size=(n_hidden, self.n_feat))
        W2 = np.random.normal(size=(n_hidden))

        b1 = np.random.normal(size=(n_hidden, 1))
        b2 = np.random.normal()

        # Gera label inicial para alguns
        l = int(n / 5)
        k = n - l
        Yl = (-1)**np.random.randint(0,2,l)

        # Cria os pesos das arestas dos demais
        adj = np.reshape(self.adj[l:,], newshape=(n*k))
        unknown = self.feats[:,(n*l):]
        temp = reLu(W1 @ unknown + b1)
        temp = np.reshape(((W2 @ temp)+b2) * adj, newshape=(k,n))
        W = 1/(1 + np.exp(-temp))

        # Matriz A
        D = np.sum(W,1) 
        invA = np.diag(1/D)

        prevY = np.zeros(k)
        self.hist = [np.concatenate((Yl, prevY))]

        # Itera para convergir
        for iter in range(100):
            Y = invA @ (W @ np.concatenate((Yl, prevY)))
            self.hist.append(np.concatenate((Yl, Y)))

            if np.linalg.norm(Y - prevY) < 0.01:
                break
            
            prevY = Y

        # Guarda Rótulos
        Ytrain = np.concatenate((Y[:self.n_train],np.zeros(self.n_unlab)))
        self.Ytrain = delimiter(Ytrain)
        self.Yl = delimiter(np.concatenate((Yl, Y))[:(self.n_lab)])



    def animate_convergence(self,real=True,save=False,name="vid.mp4"):
        """
        Cria uma animação do algoritmo de label propagation
        """

        # Cria figura
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('off')

        # Cria grafo
        H = nx.from_numpy_matrix(self.adj)
        pos = nx.spring_layout(H)
        vertices = nx.draw_networkx_nodes(
            H, pos, node_color=self.hist[0])
        arestas = nx.draw_networkx_edges(H, pos)

        # Função que atualiza o plot
        def update(n):
            y_hat = self.hist[n]
            if not real: y_hat = np.array(delimiter(y_hat))

            vertices.set_array(y_hat)
            ax.set_title("Iteração: {}".format(n), loc='left')

            return vertices,
        
        # Chama animação
        anim = FuncAnimation(
            fig, update, blit=False, interval=500, frames=len(
                self.hist))
        
        # Salva ou exibe
        if save:
            anim.save(name)
        else:
            plt.show()

