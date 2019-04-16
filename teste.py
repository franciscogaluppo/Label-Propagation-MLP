import graph
import networkx as nx
import matplotlib.pyplot as plt

G = graph.graph(10,10,10,10)
G.animate_convergence(save=True, name="Teste3.mp4")
G.animate_convergence(save=True, delimiter=True, name="Teste4.mp4")
