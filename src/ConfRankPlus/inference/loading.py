import os
import torch
from typing import Tuple, Dict, Optional


def load_ConfRankPlus(device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.float32,
                      compute_forces: bool = False) -> Tuple[torch.jit.ScriptModule, Dict[str, int]]:
    """
    :param device: torch.device
    :param dtype: dtype, default torch.float32
    :param compute_forces: Whether the model returns forces
    :param compute_stress: Wether the model returns stress
    :return: Tuple of TorchScript module and a dictionary with the fidelity mapping of the model

    The forward method of the model has the following input signature:

    def forward(
        pos: Tensor,  # [N_atoms, 3] in Angstrom
        z: Tensor,  # [N_atoms,]
        total_charge: Tensor [N_batch,]
        edge_index: Tensor,  # [2, N_edges]
        batch: Tensor,  # [N_atoms,]
        dataset_index: Tensor  # [N_atoms,]
    ) -> Dict[str, Tensor]

    The model returns a dictionary with keys:
    'energy' [kcal/mol]
    'node_energies' [kcal/mol]
    'forces' [kcal/mol/Ang]  (optional)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(os.path.dirname(__file__), "pretrained/2025_05_confrankplus.pt")
    model = torch.jit.load(path, map_location=torch.device("cpu")).eval()
    model.to(device=device, dtype=dtype)
    model.compute_forces = compute_forces
    fidelity_mapping = {"r2SCAN-3c": 0, "wB97M-D3": 1}
    return model, fidelity_mapping
