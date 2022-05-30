import networkx as nx
G = nx.Graph()
import torch
print(torch.__version__)
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root = "datas", name="PROTEINS")
print(G)