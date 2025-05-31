import torch
from torch import Tensor
from typing import Dict
from .base import BaseModel
from .layers import GNN
from .charge_model import ChargeModel
from .repulsion import RepulsionEnergy


class ConfRankPlus(BaseModel):
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        int_emb_size: int,
        out_emb_channels: int,
        pair_basis_dim: int,
        triplet_basis_dim: int,
        cutoff: float,
        cutoff_threebody: float = 0.0,
        output_dropout_p: float = 0.0,
        additive_eeq_energy: bool = True,
        additive_repulsion_energy: bool = False,
        dataset_encoding_dim: int = 0,
        num_dataset_embeddings: int = 1,
    ):
        hyperparameters = dict(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            out_emb_channels=out_emb_channels,
            pair_basis_dim=pair_basis_dim,
            triplet_basis_dim=triplet_basis_dim,
            output_dropout_p=output_dropout_p,
            cutoff=cutoff,
            cutoff_threebody=cutoff_threebody,
            additive_eeq_energy=additive_eeq_energy,
            additive_repulsion_energy=additive_repulsion_energy,
            dataset_encoding_dim=dataset_encoding_dim,
        )

        super().__init__(
            hyperparameters=hyperparameters,
            compute_forces=False,  # can be turned on manually after instantiating the model though.
            num_datasets=num_dataset_embeddings,
        )

        self.gnn = GNN(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            out_emb_channels=out_emb_channels,
            pair_basis_dim=pair_basis_dim,
            triplet_basis_dim=triplet_basis_dim,
            output_dropout_p=output_dropout_p,
            cutoff=cutoff,
            cutoff_threebody=cutoff_threebody,
            dataset_encoding_dim=dataset_encoding_dim,
            out_channels=1,
        )

        if dataset_encoding_dim > 0:
            self.dataset_encoding = torch.nn.Embedding(
                num_embeddings=num_dataset_embeddings, embedding_dim=dataset_encoding_dim
            )
        else:
            self.dataset_encoding = None

        self.charge_model = ChargeModel()
        self.additive_eeq_energy = additive_eeq_energy
        self.repulsion_model = RepulsionEnergy.gfn2() if additive_repulsion_energy else None
        self.cutoff = cutoff

    def model_forward(self, data: Dict[str, Tensor]):
        z = data["z"].long()
        pos = data["pos"]
        edge_index = data["edge_index"]
        batch = data["batch"]

        batch_size = len(torch.unique(batch))
        _total_charge = (
            data["total_charge"]
            if "total_charge" in data.keys()
            else torch.zeros(batch_size, device=batch.device, dtype=pos.dtype)
        )
        electrostatic_energy_atomwise, eeq_charges = self.charge_model(
            z, pos, _total_charge, batch
        )
        total_charge = _total_charge.view(-1, 1)

        if self.dataset_encoding is not None:
            dataset_idx = data["dataset_idx"]
            dataset_encoding = self.dataset_encoding(dataset_idx)
        else:
            dataset_encoding = None

        energy = self.gnn(
            z=z,
            pos=pos,
            edge_index=edge_index,
            batch=batch,
            eeq_charges=eeq_charges,
            total_charge=total_charge,
            dataset_encoding=dataset_encoding,
        )

        if self.additive_eeq_energy:
            electrostatic_energy = torch.sum(
                electrostatic_energy_atomwise, dim=-1, keepdim=True
            )
            energy += electrostatic_energy

        if self.repulsion_model is not None:
            repulsion_energy = self.repulsion_model(z, pos, edge_index, batch)
            energy += repulsion_energy

        return energy
