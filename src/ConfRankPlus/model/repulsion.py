import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict
from ConfRankPlus.util.scatter import scatter_add
from ConfRankPlus.util.elements import SymbolToAtomicNumber, element_symbols
from ConfRankPlus.util.units import AU2KCAL, AA2AU, EV2AU


class RepulsionEnergy(nn.Module):
    def __init__(self,
                 arep_table: Dict[int, float],
                 zeff_table: Dict[int, float],
                 kexp: float,
                 cutoff: float = 3.0):
        """
        Initializes the RepulsionEnergy module compatible with xtb by the Grimme group.

        Args:
            arep_table (dict): Mapping from atomic number to arep values.
            kexp_table (dict): Mapping from atomic number to kexp values.
            zeff_table (dict): Mapping from atomic number to zeff values.
            cutoff (float): Real-space cutoff distance.
        """
        super(RepulsionEnergy, self).__init__()

        self.cutoff = cutoff

        # Determine the maximum atomic number for table size
        supported_elements = list(set(zeff_table.keys()).union(set(arep_table.keys())))

        max_z = max(max(arep_table.keys()), max(zeff_table.keys()))
        arep = torch.zeros(max_z + 1)
        zeff = torch.zeros(max_z + 1)

        for z in arep_table:
            arep[z] = arep_table[z]
        for z in zeff_table:
            zeff[z] = zeff_table[z]

        self.AA2AU = AA2AU
        self.AU2KCAL = AU2KCAL
        self.register_buffer('arep', arep)
        self.register_buffer('kexp', torch.tensor(kexp))
        self.register_buffer('zeff', zeff)
        self.register_buffer('supported_elements', torch.tensor(supported_elements, dtype=torch.long))

    @classmethod
    def gfn1(cls):
        import os
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, "../parameters/gfn1-xtb.json")
        return cls.from_json(path)

    @classmethod
    def gfn2(cls):
        import os
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, "../parameters/gfn2-xtb.json")
        return cls.from_json(path)

    @classmethod
    def from_json(cls, path: str):
        import json
        arep_table = {}
        zeff_table = {}
        with open(path, 'r') as file:
            data = json.load(file)
            for element in element_symbols:
                try:
                    atomic_number = SymbolToAtomicNumber[element]
                    arep_table[atomic_number] = data["element"][element]["arep"]
                    zeff_table[atomic_number] = data["element"][element]["zeff"]
                except:
                    pass
            kexp = data["repulsion"]["effective"]["kexp"]
        return cls(arep_table=arep_table, zeff_table=zeff_table, kexp=kexp)

    def forward(self, z: Tensor, pos: Tensor, edge_index: Tensor, batch: Tensor):
        """
        Computes the repulsion energy for each graph in the batch.

        Args:
            z (Tensor): Atomic numbers, shape [num_atoms].
            pos (Tensor): Atom positions, shape [num_atoms, 3].
            edge_index (LongTensor): COO format edge indices, shape [2, num_edges].
            batch (LongTensor): Batch vector, assigns each atom to a graph, shape [num_atoms].

        Returns:
            Tensor: Repulsion energy per graph, shape [num_graphs].
        """

        batch_elements = torch.unique(z)

        supported_elements: List[int] = self.supported_elements.tolist()
        for el in batch_elements:
            assert el.item() in supported_elements, f"Element {el} is not supported by this module."

        src, dst = edge_index[0], edge_index[1]  # Shape: [2, num_edges]
        batch_src = batch[src]  # Shape: [num_edges]
        batch_dst = batch[dst]  # Shape: [num_edges]

        # Ensure edges are within the same graph
        same_graph = batch_src == batch_dst
        src = src[same_graph]
        dst = dst[same_graph]

        pos_src = pos[src]
        pos_dst = pos[dst]

        # Compute distances
        diff = pos_src - pos_dst
        distances = torch.clamp(torch.norm(diff, dim=1, p=2), min=1e-9)  # Prevent division by zero

        # Apply cutoff
        within_cutoff = distances <= self.cutoff
        if within_cutoff.sum() == 0:
            return torch.zeros(batch.max() + 1, device=pos.device, dtype=pos.dtype)

        src = src[within_cutoff]
        dst = dst[within_cutoff]
        distances = distances[within_cutoff] * self.AA2AU
        batch_indices = batch[src]

        # Lookup arep, kexp, zeff
        arep_src = self.arep[z[src]]
        arep_dst = self.arep[z[dst]]
        arep_term = torch.sqrt(arep_src * arep_dst)

        zeff_src = self.zeff[z[src]]
        zeff_dst = self.zeff[z[dst]]

        kexp = self.kexp.to(dtype=pos.dtype, device=pos.device)

        # Compute R_AB ** k_f
        r_k = distances.pow(kexp)

        # Compute exp(-arep * R_AB ** k_f)
        exp_term = torch.exp(-arep_term * r_k)

        # Compute repulsion energy: zeff_A * zeff_B * exp_term / R_AB
        repulsion = zeff_src * zeff_dst * exp_term / distances

        # Scatter sum per graph
        total_repulsion = scatter_add(repulsion, batch_indices, dim=0,
                                      dim_size=len(torch.unique(batch))) * self.AU2KCAL
        return total_repulsion.view(-1, 1)
