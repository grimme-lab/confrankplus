import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict
from ConfRankPlus.util.scatter import scatter_add
from ConfRankPlus.util.elements import SymbolToAtomicNumber, element_symbols
from ConfRankPlus.util.units import AU2KCAL, AA2AU, EV2AU


class ZBL_potential(torch.nn.Module):
    """ Implementation of the Ziegler-Biersack-Littmark (ZBL) potential.
    Implementation inspired from MACE implementation:
    https://github.com/ACEsuit/mace/blob/main/mace/modules/radial.py
    """

    def __init__(self, **kwargs):
        super().__init__()
        # get ZBL parameters
        self.register_buffer("a_prefactor", torch.tensor(0.4543, dtype=torch.get_default_dtype()))
        self.register_buffer("a_exp", torch.tensor(0.300, dtype=torch.get_default_dtype()))
        self.register_buffer("c", torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()))

    def forward(
            self,
            coordinates: torch.Tensor,
            species: torch.Tensor,
            edge_index: torch.Tensor,
            shifts: torch.Tensor,
            batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(torch.unique(batch))
        Z_i = species[edge_index[0]]  # shape: num_edges
        Z_j = species[edge_index[1]]  # shape: num_sdges
        a = self.a_prefactor / (torch.pow(Z_i, self.a_exp) + torch.pow(Z_j, self.a_exp))  # shape: num_edges

        difference = (coordinates[edge_index[0]]) - (coordinates[edge_index[1]] + shifts)  # shape: num_edges
        distances = torch.norm(difference, dim=1) + 1e-9  # shape: num_sdges

        r_over_a = distances / a  # shape: num_edges

        phi = (
                self.c[0] * torch.exp(-3.2 * r_over_a)
                + self.c[1] * torch.exp(-0.9423 * r_over_a)
                + self.c[2] * torch.exp(-0.4028 * r_over_a)
                + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )  # shape: num_edges

        edge_energy = (
                              14.3996 * Z_i * Z_j) / distances * phi  # e^2/(4*pi*epsilon_0) = 14.3996 eV/Angström shape: num_edges
        # reduce to num_nodes
        node_energy = scatter_add(x=edge_energy, idx_i=edge_index[0], dim_size=len(batch), dim=0)
        # reduce to batch_size
        graph_energy = scatter_add(x=node_energy, idx_i=batch, dim_size=batch_size, dim=0)

        # compute energy per graph using scatter_add
        return node_energy, graph_energy


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


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from ConfRankPlus.model.charge_model import ChargeModel
    from tad_multicharge import eeq

    # Load the repulsion energy model
    repulsion = torch.jit.script(RepulsionEnergy.gfn2())
    zbl_repulsion = ZBL_potential()
    eeq_model = ChargeModel()

    # Prepare to store distances and corresponding energies
    distances = torch.linspace(1.0, 5.0, 100)

    elements = [1, 6, 8, 16, 17, 20, 40]  # List of elements to generate pairwise combinations

    fig, axs = plt.subplots(len(elements), len(elements), figsize=(15, 15))

    for i, elem1 in enumerate(elements):
        for j, elem2 in enumerate(elements):
            if i > j:
                continue
            z = torch.tensor([elem1, elem2], dtype=torch.long)

            energies = []
            energies_eeq_list = []
            energies_eeq_og_list = []
            energies_repulsion_list = []
            energies_zbl_list = []

            # Edge index for a diatomic molecule (connecting the two atoms)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

            # Batch assignment for a single molecule
            batch = torch.tensor([0, 0], dtype=torch.long)

            # Loop over distances to calculate energies
            for d in distances:
                # Positions of the two atoms at each distance
                pos = torch.tensor([
                    [0.0, 0.0, 0.0],  # Atom 1 at the origin
                    [0.0, 0.0, d],  # Atom 2 along the z-axis at distance d
                ], dtype=torch.float)

                # Compute the repulsion energy
                energy_repulsion = repulsion(z, pos, edge_index, batch)
                total_charge = torch.tensor([-0.0], dtype=pos.dtype, device=pos.device)
                energies_eeq, charges_eeq = eeq_model(z,
                                                      pos,
                                                      total_charge,
                                                      batch)
                energy_eeq = energies_eeq.sum()
                energy_eeq_og = eeq.get_energy(z.unsqueeze(0), pos.unsqueeze(0) * AA2AU, total_charge).view(
                    -1).sum() * AU2KCAL

                # zbl:
                shifts = torch.zeros(pos.shape[0], 3, device=pos.device, dtype=pos.dtype)
                _, zbl_energy = zbl_repulsion.forward(coordinates=pos,
                                                      species=z,
                                                      edge_index=edge_index,
                                                      batch=batch,
                                                      shifts=shifts)
                zbl_energy = zbl_energy * EV2AU * AU2KCAL
                energy = energy_eeq + energy_repulsion
                energies.append(energy.item())
                energies_eeq_list.append(energy_eeq.item())
                energies_eeq_og_list.append(energy_eeq_og.item())
                energies_repulsion_list.append(energy_repulsion.item())
                energies_zbl_list.append(zbl_energy.view(-1))

            # Convert distances and energies to numpy arrays for plotting
            distances_np = distances.numpy()
            energies_np = torch.tensor(energies).numpy()

            # Plot on the subplot corresponding to this pair of elements
            # axs[i, j].plot(distances_np, energies_repulsion_list, label='Repulsion Energy')
            # axs[i, j].plot(distances_np, energies_eeq_list, label='EEQ Energy')
            # axs[i, j].plot(distances_np, energies_zbl_list, label='ZBL Energy')
            # axs[i, j].plot(distances_np, energies_eeq_og_list, label='original EEQ Energy')
            axs[i, j].plot(distances_np, energies_np, label='Total Energy')
            axs[i, j].set_xlabel('Distance (Å)')
            axs[i, j].set_ylabel('Energy (kcal/mol)')
            axs[i, j].set_title(f'Energy vs. Distance for {elem1}-{elem2} Molecule')
            axs[i, j].legend()
            axs[i, j].grid(True)

    plt.tight_layout()
    plt.show()
