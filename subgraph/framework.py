#%%
import os.path as osp
from re import sub
from sklearn import neighbors
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import copy
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GNNExplainer
from torch import Tensor
from torch_geometric.loader import DataLoader

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph
import numpy as np

import networkx as nx
from torch_geometric import utils


def expand(starts, ends, target, max_depth=1, prev=[]):
    node_neighbors = np.array([ends[idx] for idx, node in enumerate(starts) if node == target and ends[idx] != target])
    prev.append(target)
    if max_depth > 1:
        for n in node_neighbors:
            node_neighbors = np.concatenate((node_neighbors, expand(starts, ends, target=n, max_depth=max_depth-1, prev=prev)), axis=0)
    indices = np.unique(node_neighbors, return_index=True)[1]
    return np.array([node_neighbors[i] for i in sorted(indices)])

def mysubgraph(data, subset):
    out = subgraph(subset, data.edge_index, relabel_nodes=False,
                   num_nodes=data.num_nodes, return_edge_mask=True)
    edge_index, _, edge_mask = out
    if subset.dtype == torch.bool:
        num_nodes = int(subset.sum())
    else:
        num_nodes = subset.size(0) 
    _data = copy.copy(data)
    for key, value in _data:
        if key == 'edge_index':
            _data.edge_index = edge_index
        elif key == 'num_nodes':
            _data.num_nodes = num_nodes
        elif isinstance(value, Tensor):
            if data.is_node_attr(key):
                _data[key] = value[subset]
            elif data.is_edge_attr(key):
                _data[key] = value[edge_mask]
    return _data

def my_to_networkx(data, node_attrs=None, edge_attrs=None,
                to_undirected = False,
                remove_self_loops: bool = False):
    import networkx as nx

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(list(set(data.edge_index[0].numpy())))
    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, value in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(value):
            value = value if value.dim() <= 1 else value.squeeze(-1)
            values[key] = value.tolist()
        else:
            values[key] = value

    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        
        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue
        if remove_self_loops and u == v:
            continue
        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]
    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    return G

def process_one_graph(data):
    used = []
    num_nodes = data.x.shape[0]
    subgraph_sizes = [int(num_nodes/5)]
    start_nodes, end_nodes = np.array(data.edge_index)
    sub_graphs = []
    # make a grow from each node
    for target_idx in range(num_nodes):
        nodes_to_keep = expand(starts=start_nodes, ends=end_nodes, target=target_idx, max_depth=3, prev=[target_idx])
        if nodes_to_keep.shape[0] == 0:
            continue
        for size in subgraph_sizes:
            # select the grows based on size
            _subset = nodes_to_keep[:size]
            # remove repetitive stuff
            if set(_subset) in used:
                continue
            else:
                used.append(set(_subset))
            _subset = torch.from_numpy(np.array(_subset))
            sub_graphs.append((target_idx, mysubgraph(data=data, subset=_subset)))
    return sub_graphs


if __name__ == "__main__":
    dataset = 'PROTEINS'
    path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data', 'TUDataset')
    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = TUDataset(path, dataset, transform=transform)
    dataset = dataset[:1]
    
    all_subgraphs = []
    for data in tqdm.tqdm(dataset):
        subgraphs = process_one_graph(data)
        all_subgraphs.extend(subgraphs)

    for target_idx, graph in tqdm.tqdm(all_subgraphs):
        nx.draw_networkx(my_to_networkx(graph, remove_self_loops=True))
        plt.show()
        
        
        
# %%
