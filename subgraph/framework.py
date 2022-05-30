#%%
import os.path as osp
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GNNExplainer

from torch_geometric.loader import DataLoader

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph
import numpy as np

dataset = 'PROTEINS'
path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data', 'TUDataset')
transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
dataset = TUDataset(path, dataset, transform=transform)

import networkx as nx
from torch_geometric import utils


# %%
data = dataset[0]
_subset = torch.from_numpy(np.array(list(range(20))))
sub_data = data.subgraph(subset=_subset)
nx.draw_networkx(utils.to_networkx(data, remove_self_loops=True))


# %%
nx.draw_networkx(utils.to_networkx(sub_data, remove_self_loops=True))


# %%