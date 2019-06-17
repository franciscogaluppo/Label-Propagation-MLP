import graph import graph
from operator import itemgetter as get
from itertools import groupby as gb

direct = lambda x: x if x[0] <= x[1] else x[::-1]
direct_list = lambda x: [direct(a) for a in x]

class read_graph_vertex_features(graph):

    def __init__(self, edge_list, vertex_features, n_lab, n_tar, labels, direcionado=True):
    """
    Inicializa classe
    :param edge_list: lista das arestas do grafo.
    :param vextex_features: lista de features dos vértices 
    :param n_lab: número de vértices no treino
    :param n_tar: número de vértices na validação
    :param labels: lista de labels dos vértices
    """    

    # Gurda as arestas ordenadas
    if direcionado:
        self.edges = len(edge_list)
        self.edge_list = sorted(edge_list, get(1, 2))
    
    else:
        self.edge_list = [x for x,_ in gb(sorted(direct_list(edge_list), get(1, 2)))]
        self.edges = int(sum(self.edge_list) / 2)

    # Salva valores para que os outros métodos possam acessar
    self.vertices = len(vertex_features)
    self.vertex_features = vertex_features
    self.n_lab = n_lab
    self.n_tar = n_tar
    self.labels = labels
    self.direcionado = direcionado

    # Chama os métodos de criação
    self.create_features()
    self.create_arrays()


    def create_features(self):
    """
    Combinas as features dos vértices e cria as features para todo par de vértices.
    A feature é de zeros quando não existe
    """
    
    if not self.direcionado:
        raise Exception("Ainda não implementado para grafos não direcionados.")

    n = self.vertices
    n_feat = 2*len(self.vertex_features[0])

    # Cria features e as coloca no formato padrão
    feats = np.zeros((n*n, n_feat))
    for e in range(self.edges):
        i, j = self.edge_list[e]
        feats[i*n+j] = np.array(self.vertex_features[i] + self.vertex_fetures[j])
    self.feats = feats.T


    def create_arrays(self):
    """
    Define Ylabel, Ytarget e Ytest
    """
    
    self.Ylabel = labels[:self.n_lab,:]
    self.Ytarget = labels[self.n_lab:(self.n_lab + self.n_tar)]
    self.Ytest = labels[(self.n_lab+n_tar)-n:,:]
