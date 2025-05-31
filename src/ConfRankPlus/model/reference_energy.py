import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from typing import Optional, Dict, Callable
from tqdm import tqdm
from warnings import warn
import numpy as np
from typing import List
from ..util.scatter import scatter_add


class ReferenceEnergies(torch.nn.Module):
    """
    PyTorch module for fitting the reference energies (per species) of given datasets
    """

    __version__ = 1

    def __init__(
            self,
            freeze: bool = True,
            num_embeddings: int = 104,
            max_num_datasets: int = 1,
    ) -> None:
        """
        :param freeze: Freeze module weights after init, default: True -> No optimization during training
        :param target_key: Name of the target, default: 'energy'
        :param num_embeddings: Number of expected elements, default: default
        :param max_num_datasets: Maximal number of datasets that this class should be able to handle
        """
        super(ReferenceEnergies, self).__init__()
        # constant energy shifts for all datasets stored in an array
        self.constant_shifts = torch.nn.Parameter(
            torch.zeros(num_embeddings, max_num_datasets), requires_grad=bool(~freeze)
        )

    @property
    def num_embeddings(self) -> int:
        return self.constant_shifts.shape[0]

    @num_embeddings.setter
    def num_embeddings(self, val):
        warn(
            "Cannot change number of embeddings after instance has been created. Create a new instance!"
        )

    @property
    def num_datasets(self) -> int:
        return self.constant_shifts.shape[1]

    @num_datasets.setter
    def num_datasets(self, val):
        warn(
            "Cannot change number of datasets after instance has been created. Create a new instance!"
        )

    def fit_constant_energies(
            self,
            trainset: Dataset,
            target_key: str = "energy",
            freeze_after: bool = True,
            atomic_numbers: List[int] = None,
    ) -> None:
        """
        Fit the reference energies against the number of atoms in each structure
        using a direct linear solution.

        :param trainset: Dataset or Concatenation of datasets
        :param target_key: Key to access target energy values
        :param freeze_after: Whether the weights/reference energies should be frozen after this fit
        :param atomic_numbers: If provided this speed up the code a bit
        :return: None
        """

        if atomic_numbers is None:
            unique_atomic_numbers = set()
            temp_loader = DataLoader(dataset=trainset, batch_size=500, num_workers=8)
            for data in tqdm(temp_loader, desc="Find unique elements in dataset"):
                unique_atomic_numbers.update(torch.unique(data["z"]).tolist())
        else:
            unique_atomic_numbers = atomic_numbers

        num_elements = len(unique_atomic_numbers)
        element_to_index = {
            element: index for index, element in enumerate(unique_atomic_numbers)
        }

        A_dict = {dataset_idx: [] for dataset_idx in range(self.num_datasets)}
        b_dict = {dataset_idx: [] for dataset_idx in range(self.num_datasets)}
        for i, data in enumerate(tqdm(trainset, desc="Setup matrix for linear fit.")):
            dataset_idx = (
                data["dataset_idx"].item() if "dataset_index" in data.keys() else 0
            )
            assert (
                    dataset_idx in A_dict.keys()
            ), f"Found dataset_index {dataset_idx} but allow only {A_dict.keys()}"
            atomic_numbers, counts = torch.unique(data["z"], return_counts=True)
            A_entry = np.zeros(num_elements)
            for an, c in zip(atomic_numbers, counts):
                idx = element_to_index[an.item()]
                A_entry[idx] = c.item()
            A_dict[dataset_idx].append(A_entry)
            b_dict[dataset_idx].append(data[target_key].numpy())

        for dataset_idx in range(self.num_datasets):
            if len(A_dict[dataset_idx]) == 0:
                continue
            A = np.vstack(A_dict[dataset_idx])
            b = np.stack(b_dict[dataset_idx]).flatten()
            assert A.shape[0] == b.shape[0]

            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            if rank < min(*A.shape):
                warn(
                    f"Coefficient matrix does not have full rank. Be careful with respect to the resulting isolated atom energies!"
                )

            atom_energy_dict = {
                element: energy for element, energy in zip(element_to_index.keys(), x)
            }

            print(
                f"Isolated atom energies for dataset {dataset_idx} as obtained by linear fit:",
                atom_energy_dict,
            )
            self.set_constant_energies(
                energy_dict=atom_energy_dict,
                dataset_index=dataset_idx,
                freeze=freeze_after,
            )

    def set_constant_energies(
            self,
            energy_dict: Dict[int, float],
            dataset_index: int = 0,
            freeze: bool = True,
    ) -> None:
        """
        :param energy_dict: Mapping of {atomic_number : reference_energy}
        :param dataset_index: If there are more one datasets, this might be useful, default: 0
        :param freeze: Should the model weights be frozen after setting the energies?, default: True
        :return: None
        """
        assert dataset_index < self.constant_shifts.shape[1]

        self.constant_shifts.requires_grad_(False)
        for z in energy_dict.keys():
            self.constant_shifts[z][dataset_index] = energy_dict[z]
        self.constant_shifts.requires_grad_(freeze)

    def forward(
            self,
            species: Tensor,
            batch: Tensor,
            dataset_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param species: 1D Tensor with atomic numbers of shape (N,)
        :param batch: 1D Tensor with batch indices of shape (N,)
        :param dataset_index: 1D Tensor with index of the dataset for each atom -> shape (N,)
        :return: Contribution from summing up reference energies
        """
        if dataset_index is not None:
            energy_shift_per_atom = self.constant_shifts[species, dataset_index]
        else:
            # if no dataset_index is specified, just use the first column of self.constant_shifts
            energy_shift_per_atom = self.constant_shifts[..., 0][species]

        energy = scatter_add(
            energy_shift_per_atom, dim=0, dim_size=len(torch.unique(batch)), idx_i=batch
        )

        return energy
