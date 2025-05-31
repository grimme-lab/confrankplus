from __future__ import annotations

from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.transforms import BaseTransform


class HDF5Dataset(InMemoryDataset):

    def __init__(
            self,
            data: BaseData,
            slices: Dict[str, Tensor],
            transform: Optional[BaseTransform] = None,
    ):
        super().__init__("./", transform)
        self.data, self.slices = data, slices

    @staticmethod
    def from_hdf5(
            filepath: str,
            precision: int = 32,
            transform: Optional[BaseTransform] = None,
            exclude_keys: Optional[List[str]] = None,
    ) -> HDF5Dataset:
        _exclude_keys = exclude_keys if exclude_keys is not None else []
        data = {}
        slices = {}
        with h5py.File(filepath, "r") as f:
            if "additional" in f.keys() and "utf-8-encoded" in f["additional"].keys():
                decode_utf8 = [
                    key.decode("utf-8") for key in f["additional"]["utf-8-encoded"][:]
                ]
            else:
                decode_utf8 = []
            for key in f["data"].keys():
                np_arrays = {"data": f["data"][key][:], "slices": f["slices"][key][:]}
                for prop, val in np_arrays.items():
                    if val.dtype == np.uint64:
                        np_arrays[prop] = val.astype(np.int64)
                    if val.dtype == np.float64 and precision == 32:
                        np_arrays[prop] = val.astype(np.float32)
                    elif val.dtype == np.float32 and precision == 64:
                        np_arrays[prop] = val.astype(np.float64)
                if key in decode_utf8:
                    data[key] = np_arrays["data"].tolist()
                    data[key] = [bs.decode("utf-8") for bs in data[key]]
                    slices[key] = torch.from_numpy(np_arrays["slices"])
                else:
                    data[key] = torch.from_numpy(np_arrays["data"])
                    slices[key] = torch.from_numpy(np_arrays["slices"])
        dataset = HDF5Dataset(
            data=Data.from_dict(data),
            slices=slices,
            transform=transform,
        )
        return dataset

    @staticmethod
    def from_ase(
            filepaths: List[str],
            file_format: Optional[str] = None,
            precision: int = 32,
            index: Optional[slice] = ':',
            transform: Optional[BaseTransform] = None,
    ) -> HDF5Dataset:

        from ase.io import read
        assert precision in [32, 64], 'Precision must be either 32 or 64!'

        data = {}
        slices = {}
        dtype = torch.double if precision == 64 else torch.float32

        for filepath in filepaths:
            atoms_list = read(filepath, index=index, format=file_format)
            for i, atoms in enumerate(atoms_list):
                atom_data = {
                    "pos": torch.tensor(atoms.positions, dtype=dtype),
                    "z": torch.tensor(atoms.numbers, dtype=torch.long).view(-1),
                }

                for key, value in atom_data.items():
                    if key not in data:
                        data[key] = []
                        slices[key] = [0]
                    data[key].append(value)
                    increment_slice = value.shape[0]
                    slice_index = slices[key][-1] + increment_slice
                    slices[key].append(slice_index)

        for key in data:
            data[key] = torch.cat(data[key], dim=0)
            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        dataset = HDF5Dataset(
            data=Data.from_dict(data),
            slices=slices,
            transform=transform,
        )
        return dataset
