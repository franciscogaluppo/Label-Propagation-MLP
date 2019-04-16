import graph
import networkx as nx
import matplotlib.pyplot as plt

G = graph.graph(7,7,7,30)
H = nx.from_numpy_matrix(G.adj)
print(G.y_hat_not_real)
nx.draw(H, with_labels=True, node_color=G.y_hat_not_real)
plt.show()
