from dgl import DGLGraph
src=[0,0,1,2,3,4]
dst=[0,2,3,2,3,0]
g = DGLGraph((src, dst))

import matplotlib.pyplot as plt
import networkx as nx
nx.draw(g.to_networkx(), with_labels=True)
plt.show()