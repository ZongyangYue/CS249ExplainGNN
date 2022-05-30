#%%
import os.path as osp
from re import sub
from sklearn import neighbors
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


def process_one_graph(data):
    used = []
    num_nodes = data.x.shape[0]
    subgraph_sizes = [int(num_nodes/n) for n in range(2, 5)]
    start_nodes, end_nodes = np.array(data.edge_index)
    sub_graphs = []
    # make a grow from each node
    for target_idx in range(num_nodes):
        nodes_to_keep = expand(starts=start_nodes, ends=end_nodes, target=target_idx, max_depth=3, prev=[])
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
            sub_graphs.append((target_idx, data.subgraph(subset=_subset)))
    return sub_graphs


if __name__ == "__main__":
    dataset = 'PROTEINS'
    path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data', 'TUDataset')
    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = TUDataset(path, dataset, transform=transform)

    all_subgraphs = []
    for data in tqdm.tqdm(dataset):
        subgraphs = process_one_graph(data)
        all_subgraphs.extend(subgraphs)
    
    for target_idx, graph in tqdm.tqdm(all_subgraphs):
        nx.draw_networkx(utils.to_networkx(graph, remove_self_loops=True))
        plt.show()
