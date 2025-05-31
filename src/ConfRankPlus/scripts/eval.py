import argparse
import logging
import os
from typing import List, Optional, Union

import torch
from tqdm import tqdm
from ase import Atoms
from ase.io import write
from torch_geometric.loader import DataLoader
from ConfRankPlus.inference.radius_graph import SmartRadiusGraph
from ConfRankPlus.data.dataset import HDF5Dataset
from ConfRankPlus.inference.loading import load_ConfRankPlus


def to_ase_atoms(data, properties, fidelity, write_additional_energies: bool = False):
    positions = data['pos']
    numbers = data['z']
    total_charge = data['total_charge']
    energy = properties.get('energy', None)
    forces = properties.get('forces', None)
    ensbid = data.get('ensbid', None)
    confid = data.get('confid', None)

    additional_data = {}
    ENERGY_KEY = f'confrankplus_energy_{fidelity}'
    FORCES_KEY = f'confrankplus_forces_{fidelity}'

    for key in data.keys():
        if "energy" in key and key != ENERGY_KEY:
            additional_data[key] = data[key]

    batch_idx_unique = torch.unique(data["batch"]).tolist()
    atoms_list = []

    for i, batch_idx in enumerate(batch_idx_unique):

        batch_mask = (data["batch"] == batch_idx).cpu()
        atoms = Atoms(numbers=numbers[batch_mask].detach().cpu().numpy(),
                      positions=positions[batch_mask].detach().cpu().numpy())

        if ensbid is not None:
            atoms.info['ensbid'] = str(ensbid[i])
        if confid is not None:
            atoms.info['confid'] = str(confid[i])

        atoms.info['total_charge'] = total_charge[i].item()

        if energy is not None:
            atoms.info[ENERGY_KEY] = energy[i].item()
        if forces is not None:
            atoms.arrays[FORCES_KEY] = forces[batch_mask].numpy()

        if write_additional_energies:
            for key, val in additional_data.items():
                if isinstance(val, torch.Tensor):
                    _val = val[i].item()
                else:
                    continue
                atoms.info[key] = _val

        atoms_list.append(atoms)

    return atoms_list


def process_files(model: Union[torch.nn.Module, torch.jit.ScriptModule],
                  filepaths: List[str],
                  output_path: str = "confrank_output.extxyz",
                  fidelity_index: int = 0,
                  fidelity="r2SCAN-3c",
                  batch_size: int = 10,
                  device: str = None,
                  precision: int = 32,
                  num_workers: int = 1,
                  total_charge: Optional[int] = None,
                  verbose: bool = False,
                  write_additional_energies: bool = False) -> None:
    # Check if file already exists
    if os.path.exists(output_path):
        logging.warning(f'File {output_path} already exists, atoms will be appended.')

    if device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = torch.device(device)

    if verbose:
        logging.info(f'Using device={_device}, precision={precision}, batch_size={batch_size}')
        logging.info(f'Loading data...')

    model.to(_device)
    radius_graph_transform = SmartRadiusGraph(radius=model.cutoff)

    extensions = set(os.path.splitext(filepath)[1] for filepath in filepaths)
    assert len(extensions) == 1, "Not all files have the same extension"

    if list(extensions)[0] in ['.h5', '.hdf5']:
        assert len(filepaths) == 1, "Currently, only a single HDF5 file is supported as an input."
        dataset = HDF5Dataset.from_hdf5(filepaths[0],
                                        precision=precision,
                                        transform=radius_graph_transform)
    else:
        dataset = HDF5Dataset.from_ase(filepaths,
                                       precision=precision,
                                       transform=radius_graph_transform)

    if verbose:
        logging.info(f'Loaded data with {len(dataset)} samples.')

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False)
    atoms_list = []
    try:
        for data in tqdm(loader, desc="Evaluate model on input data."):
            data.to(_device)
            # handle total_charge
            if "total_charge" not in data:
                if total_charge is None:
                    _total_charge = 0
                else:
                    _total_charge = total_charge
                bs = len(data)
                data.update({"total_charge": torch.full(size=(bs,),
                                                        device=_device,
                                                        dtype=torch.long,
                                                        fill_value=int(_total_charge))})
            # predict:
            with torch.jit.optimized_execution(False):
                model_input_dict = dict(pos=data['pos'],
                                        z=data['z'].long(),
                                        edge_index=data['edge_index'],
                                        total_charge=data['total_charge'],
                                        batch=data['batch'],
                                        dataset_idx=torch.full_like(data["z"], dtype=torch.long,
                                                                    fill_value=fidelity_index))
                predictions = model.forward(model_input_dict)
                predictions = {key: val.detach().cpu() if isinstance(val, torch.Tensor) else val for key, val in
                               predictions.items()}

            batch_atoms_list = to_ase_atoms(data=data,
                                            properties=predictions,
                                            fidelity=fidelity,
                                            write_additional_energies=write_additional_energies)
            atoms_list.extend(batch_atoms_list)
    except Exception as e:
        print(e)
    finally:
        with open(output_path, 'a') as f:
            write(f, atoms_list, append=True)


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Label molecules in ase-compatible files with the ConfRankPlus model.')
    parser.add_argument('--files',
                        metavar='F',
                        type=str,
                        nargs='+',
                        help='Input files (xyz recommended) to process. '
                             'A single file can store multiple conformations. '
                             'The content of all files will be concatenated '
                             'and the model will run batched inference on it.')
    parser.add_argument('--output_path',
                        type=str,
                        default='confrank_output.extxyz',
                        help='Output filepath. Extension is used by ase for inferring output type. Extxyz recommended.'
                             ' Default: confrank_output.extxyz')
    parser.add_argument('--total_charge',
                        type=int,
                        required=False,
                        default=None,
                        help='Total charge that will be assumed for the input structures. Default: 0')
    parser.add_argument('--fidelity',
                        type=str,
                        default="wB97M-D3",
                        choices=["r2SCAN-3c", "wB97M-D3"],
                        help='Fidelity channel that will be used for inference. '
                             'Either "r2SCAN-3c" or "wB97-D3". Default: wB97M-D3')
    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        help='Batch size for inference. This will increase the demand of RAM. Default: 10')
    parser.add_argument('--device',
                        type=str,
                        required=False,
                        choices=['cuda', 'cpu'],
                        help='Device that is used for inference. '
                             'If none is provided, the code will try to run inference on GPU if possible. '
                             'If you want to enforce inference on CPU pass "cpu". ')
    parser.add_argument('--precision',
                        type=int,
                        required=False,
                        default=32,
                        choices=[32, 64],
                        help='Numerical precision for inference. Either 32-bit or 64-bit precision is possible.'
                             ' Default: 32')
    parser.add_argument('--compute_forces',
                        type=lambda x: x.lower() == 'true',
                        required=False,
                        default=False,
                        help='If forces should be computed or not. Default: False')
    parser.add_argument('--write_additional_energies',
                        type=lambda x: x.lower() == 'true',
                        required=False,
                        default=False,
                        help='If existing energies that are present in the input files should be written to the output.'
                             ' They need to have the string "energy" in their key.')

    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(device=args.device)
    else:
        device = None

    if args.precision == 64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    model, fidelity_mapping = load_ConfRankPlus(device=device,
                                                dtype=dtype,
                                                compute_forces=args.compute_forces)
    logging.info(f"Set model to fidelity={args.fidelity}")
    fidelity_index = fidelity_mapping[args.fidelity]
    process_files(model=model,
                  filepaths=args.files,
                  output_path=args.output_path,
                  fidelity_index=fidelity_index,
                  fidelity=args.fidelity,
                  batch_size=args.batch_size,
                  device=device,
                  precision=args.precision,
                  total_charge=args.total_charge,
                  verbose=True,
                  write_additional_energies=args.write_additional_energies)


if __name__ == "__main__":
    main()
