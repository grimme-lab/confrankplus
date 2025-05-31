from __future__ import annotations

from typing import Optional, Union, Literal
from warnings import warn
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ConfRankPlus.inference.radius_graph import SmartRadiusGraph
from ConfRankPlus.inference.loading import load_ConfRankPlus
from ConfRankPlus.util.units import AU2KCAL, EV2AU


class ConfRankPlusCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self,
                 model: Union[torch.nn.Module, torch.jit.ScriptModule],
                 fidelity_index: int = 0,
                 compute_forces: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32,
                 **kwargs):

        super().__init__(**kwargs)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fidelity_index = fidelity_index
        self.dtype = dtype
        self.device = device
        self.model = model.eval().to(device=device, dtype=dtype)
        self.cutoff = model.cutoff
        self._charge = 0
        if compute_forces:
            self.model.compute_forces = True
            self.implemented_properties.append("forces")
        self.compute_forces = compute_forces
        self._jit_optimize = False
        self.radius_graph_transform = SmartRadiusGraph(radius=model.cutoff)

    @classmethod
    def load_default(cls, fidelity: Literal["wB97M-D3", "r2SCAN-3c"] = "wB97M-D3",
                     device: Optional[Literal["cuda", "cpu"]] = None,
                     dtype=torch.float32,
                     compute_forces: bool = True):

        if device is not None:
            _device = torch.device(device)
        else:
            _device = None

        model, fidelity_mapping = load_ConfRankPlus(device=_device,
                                                    dtype=dtype,
                                                    compute_forces=compute_forces)
        fidelity_index = fidelity_mapping[fidelity]

        return cls(model=model,
                   fidelity_index=fidelity_index,
                   compute_forces=compute_forces,
                   device=device,
                   dtype=dtype)

    @classmethod
    def from_torchscript(cls, path: str, **kwargs):
        model = torch.jit.load(path)
        return cls(model, **kwargs)

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, charge: int):
        if charge not in [-2, -1, 0, 1, 2]:
            warn(f"The behaviour for charge={charge} has not been tested.")
        self._charge = charge

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):

        pbc = atoms.get_pbc()
        assert not np.any(pbc), "PBC ist not supported!"
        super().calculate(atoms, properties, system_changes)

        pos = torch.tensor(
            atoms.get_positions(), dtype=self.dtype
        )
        z = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long
        )
        total_charge = torch.tensor([self.charge], dtype=torch.long)
        dataset_idx = torch.full_like(z, fill_value=self.fidelity_index)

        data_dict = dict(pos=pos,  # coordinates: [N_atoms, 3]
                         z=z,  # atomic numbers: [N_atoms, ]
                         total_charge=total_charge,  # total charge: [N_batch,]
                         batch=torch.zeros_like(z),  # batch index: [N_atoms,]
                         dataset_idx=dataset_idx)  # dataset index: [N_atoms,]

        data_dict = self.radius_graph_transform(data_dict)

        data_dict = {key: val.to(self.device) for key, val in data_dict.items()}

        with torch.jit.optimized_execution(self._jit_optimize):
            prediction = self.model(data_dict)

        if "energy" not in prediction.keys():
            raise ValueError("The model did not return 'energy' in its output.")
        else:
            energy = prediction.get("energy") * (1.0 / (AU2KCAL * EV2AU))
            self.results["energy"] = energy.item()
        if "forces" in properties:
            if "forces" not in prediction.keys():
                raise ValueError("The model did not return 'forces' in its output.")
            else:
                forces = prediction.get("forces") * (1.0 / (AU2KCAL * EV2AU))
                self.results["forces"] = forces.detach().cpu().numpy()
