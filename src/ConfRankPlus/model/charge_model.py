import torch
from ..util.units import AA2AU, AU2KCAL
from .eeq import EEQModel, cn_eeq, pack


class ChargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eeq_model = EEQModel.param2019()
        self.AA2AU = AA2AU
        self.AU2KCAL = AU2KCAL

    def forward(self, z, pos, total_charge, batch):
        unique_batches = torch.unique(batch)
        n_atoms = [torch.sum(batch == batch_idx).item() for batch_idx in unique_batches]
        positions = [pos[batch == batch_idx] for batch_idx in unique_batches]
        numbers = [z[batch == batch_idx].long() for batch_idx in unique_batches]

        numbers = pack(numbers)
        positions = pack(positions) * self.AA2AU  # to bohr

        cn = cn_eeq(
            numbers,
            positions,
        )
        electrostatic_energy, eeq_charges = self.eeq_model.forward(
            numbers, positions, total_charge, cn
        )
        electrostatic_energy = self.AU2KCAL * electrostatic_energy  # to kcal/mol
        all_eeq_charges = [eeq_charges[i][: n_atoms[i]] for i in range(len(n_atoms))]
        all_eeq_charges = torch.cat(all_eeq_charges).view(-1, 1)
        return electrostatic_energy, all_eeq_charges
