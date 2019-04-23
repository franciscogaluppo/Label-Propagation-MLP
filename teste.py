import graph
import networkx as nx
import matplotlib.pyplot as plt

G = graph.graph(10,10,10,10)
G.animate_convergence(save=True, name="Teste_sem_peso_1.mp4")
G.animate_convergence(save=True, delimiter=True, name="Teste_sem_peso2.mp4")
