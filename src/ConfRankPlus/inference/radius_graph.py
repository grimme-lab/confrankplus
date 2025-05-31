import numpy as np
import torch
from torch import Tensor
from typing import Dict, Union
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

try:
    from torch_cluster import radius_graph

    _HAS_TORCH_CLUSTER = True
except ImportError:
    _HAS_TORCH_CLUSTER = False

try:
    from ase.neighborlist import primitive_neighbor_list

    _HAS_ASE = True
except ImportError:
    _HAS_ASE = False


class SmartRadiusGraph(BaseTransform):

    def __init__(self, radius: float):
        super(SmartRadiusGraph, self).__init__()
        self.radius = radius
        if _HAS_TORCH_CLUSTER:
            graph_backend = "torch_cluster"
        elif _HAS_ASE:
            graph_backend = "ASE"
        else:
            raise Exception("Need either ASE or torch_cluster for computing neighborlists.")

        print(f"Using {graph_backend} for computing neighborlists.")

    def __call__(self, data: Union[Data, Dict[str, Tensor]]) -> Data:
        if _HAS_TORCH_CLUSTER:
            edge_index = radius_graph(x=data["pos"],
                                      r=self.radius,
                                      max_num_neighbors=320,
                                      batch=data["batch"] if "batch" in data.keys() else None)
            data["edge_index"] = edge_index
            return data

        initial_device = data["pos"].device
        pos_np = data["pos"].detach().cpu().numpy()
        max_pos = max(np.max(np.linalg.norm(pos_np, axis=1)), 1e-3)
        cell = np.eye(3, 3) * (2.0 * max_pos)
        pbc = [False, False, False]

        i_indices, j_indices = primitive_neighbor_list(
            quantities="ij",
            pbc=pbc,
            cell=cell,
            positions=pos_np,
            cutoff=self.radius,
            self_interaction=False
        )
        edge_index_np = np.vstack([i_indices, j_indices])
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=initial_device)
        data["edge_index"] = edge_index
        return data
