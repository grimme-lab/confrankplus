from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Embedding, Linear, Dropout
from torch.nn import Sequential, SiLU
from .atomic_data import electronegativities, single_radii_dict
from ..util.scatter import scatter_add

def triplets_sparse(edge_index: Tensor, n_atoms: int) -> Tuple[Tensor, Tensor]:
    """
    Requires edges to be undirected
    """
    row, col = edge_index[0], edge_index[1]
    value = torch.arange(row.size(0), device=row.device)
    adj_t = torch.sparse_coo_tensor(
        indices=edge_index, values=value, size=(n_atoms, n_atoms)
    )
    adj_t_row = adj_t.index_select(dim=0, index=row)
    s_value = adj_t_row._values()
    if s_value is not None:
        idx_jk = s_value
    else:
        raise Exception
    s_row = adj_t_row._indices()[0]
    if s_row is not None:
        idx_ji = s_row
    else:
        raise Exception
    return idx_ji, idx_jk


def sqrt(x: Tensor, eps: float = 1e-8):
    out = torch.sqrt(torch.clamp(x, min=eps))
    return out


def l2norm(x: Tensor, dim: int = -1):
    x_squared = torch.sum(torch.square(x), dim=dim)
    norm = sqrt(x_squared)
    return norm


class Triplets(torch.nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(
            self,
            vec: Tensor,
            edge_index: Tensor,
            n_atoms: int,
    ) -> Tuple[Tensor, Tensor]:
        vec = vec.clone().detach()
        n_bond = vec.shape[0]
        dist = torch.norm(vec, dim=1, p=2)
        valid_three_body = dist <= self.cutoff
        original_index = torch.arange(n_bond, dtype=torch.long, device=vec.device)[
            valid_three_body
        ]
        valid_edge_index = edge_index[:, valid_three_body]
        idx_ij, idx_kj = triplets_sparse(edge_index=valid_edge_index, n_atoms=n_atoms)
        idx_ij = original_index[idx_ij]
        idx_kj = original_index[idx_kj]
        mask = ~(idx_ij == idx_kj)
        idx_ij = idx_ij[mask]
        idx_kj = idx_kj[mask]
        return idx_ij, idx_kj


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
            SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(x)


class Taper(torch.nn.Module):
    def __init__(self, cutoff: float):
        self.cutoff = cutoff
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        mask = x > self.cutoff
        x = x * (1.0 / self.cutoff)
        x3 = x * x * x
        x4 = x3 * x
        x5 = x4 * x
        out = 1 - 6 * x5 + 15 * x4 - 10 * x3
        out[mask] = out[mask] * 0.0
        return out


class PairBasis(torch.nn.Module):
    def __init__(
            self,
            output_dim: int,
            edge_feature_dim: int,
            hidden_dim: int,
            cutoff: float,
    ):
        super().__init__()

        self.taper = Taper(cutoff=cutoff)

        self.basis_network = Sequential(
            Linear(1 + edge_feature_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, output_dim),
            SiLU(),
        )

    def forward(self, dist: Tensor, edge_features: Tensor) -> Tensor:
        taper = self.taper(dist.unsqueeze(-1))

        # prepare input for nn
        inp = torch.cat(
            [
                dist.unsqueeze(1),
                edge_features,
            ],
            dim=1,
        )
        out = self.basis_network(inp)
        out = out * taper
        return out


class TripletBasis(torch.nn.Module):
    def __init__(
            self,
            output_dim: int,
            edge_feature_dim: int,
            hidden_dim: int,
            cutoff: float,
    ):
        super().__init__()

        self.taper = Taper(cutoff=cutoff)

        self.basis_network_angle = Sequential(
            Linear(1, hidden_dim), SiLU(), Linear(hidden_dim, output_dim), SiLU()
        )

        self.basis_network_radial = Sequential(
            Linear(1 + edge_feature_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, output_dim),
            SiLU(),
        )

        self.readout_network = Sequential(
            Linear(output_dim, output_dim),
            SiLU(),
            Linear(output_dim, output_dim),
            SiLU(),
        )

    def forward(
            self,
            dist: Tensor,
            angle: Tensor,
            idx_kj: Tensor,
            idx_ji: Tensor,
            edge_features: Tensor,
    ) -> Tensor:
        taper = self.taper(dist)
        angle_basis = self.basis_network_angle(angle.unsqueeze(1))
        radial_input = torch.cat([dist.unsqueeze(1), edge_features], dim=1)
        radial_basis = self.basis_network_radial(radial_input) * taper.unsqueeze(1)
        radial_basis = radial_basis[idx_ji] + radial_basis[idx_kj]
        out = angle_basis * radial_basis
        out = self.readout_network(out)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(
            self,
            output_dim: int,
            hidden_channels: int,
    ):
        super().__init__()

        self.emb = Embedding(105, hidden_channels)
        self.pair_network = Sequential(Linear(output_dim, hidden_channels), SiLU())
        self.embedding_network = Sequential(
            Linear(3 * hidden_channels, hidden_channels), SiLU()
        )

    def forward(
            self,
            x: Tensor,
            pair_basis: Tensor,
            i: Tensor,
            j: Tensor,
    ) -> Tensor:
        x = self.emb(x)
        pair_basis = self.pair_network(pair_basis)
        inp = torch.cat([x[i], x[j], pair_basis], dim=-1)
        emb = self.embedding_network(inp)
        return emb


class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            int_emb_size: int,
            pair_basis_dim: int,
            triplet_basis_dim: int,
            num_before_skip: int,
            num_after_skip: int,
    ):
        super().__init__()

        self.pair_network_initial = Sequential(
            Linear(pair_basis_dim, hidden_channels, bias=False), SiLU()
        )
        self.triplet_network_initial = Sequential(
            Linear(triplet_basis_dim, int_emb_size, bias=False), SiLU()
        )

        # Hidden transformation of input message:
        self.emb_kj = Sequential(Linear(hidden_channels, hidden_channels), SiLU())
        self.emb_ji = Sequential(Linear(hidden_channels, hidden_channels), SiLU())

        # Embedding projections for interaction:
        self.pair_project_down = Sequential(
            Linear(hidden_channels, int_emb_size, bias=False), SiLU()
        )
        self.pair_project_up = Sequential(
            Linear(int_emb_size, hidden_channels, bias=False), SiLU()
        )

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels) for _ in range(num_before_skip)]
        )

        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels) for _ in range(num_after_skip)]
        )

    def forward(
            self,
            x: Tensor,
            pair_basis: Tensor,
            triplet_basis: Tensor,
            idx_kj: Tensor,
            idx_ji: Tensor,
    ) -> Tensor:
        pair_basis = self.pair_network_initial(pair_basis)
        triplet_basis = self.triplet_network_initial(triplet_basis)

        # Initial transformation:
        x_ji = self.emb_ji(x)
        x_kj = self.emb_kj(x)

        # triplet interaction
        x_kj = x_kj * pair_basis
        x_kj = self.pair_project_down(x_kj)
        x_kj = x_kj[idx_kj] * triplet_basis

        x_kj = scatter_add(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.pair_project_up(x_kj)

        h = x_ji + x_kj

        for layer in self.layers_before_skip:
            h = layer(h)

        # skip connection
        h = h + x

        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(torch.nn.Module):
    def __init__(
            self,
            pair_basis_dim: int,
            hidden_channels: int,
            out_emb_channels: int,
            out_channels: int,
            dropout_p: float = 0.0,
    ):
        super().__init__()

        self.lin_pair = Linear(pair_basis_dim, hidden_channels, bias=False)
        self.out_network = Sequential(
            Dropout(p=dropout_p),
            Linear(hidden_channels, out_emb_channels, bias=False),
            SiLU(),
            Dropout(p=dropout_p),
            Linear(out_emb_channels, out_emb_channels),
            SiLU(),
            Linear(out_emb_channels, out_channels, bias=False),
        )

    def forward(
            self, x: Tensor, pair_basis: Tensor, i: Tensor, num_nodes: int
    ) -> Tensor:
        x = self.lin_pair(pair_basis) * x
        x = scatter_add(x, i, dim=0, dim_size=num_nodes)
        out = self.out_network(x)
        return out


class CoordinationNumberEdges(torch.nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        # Initialize Embedding layers with electronegativities and radii
        en_tensor = torch.tensor(
            [electronegativities[el] for el in range(104)]
        ).reshape(-1, 1)
        self.electronegativity_embedding = torch.nn.Embedding.from_pretrained(
            embeddings=en_tensor, freeze=True
        )

        embeddings = torch.tensor([single_radii_dict[el] for el in range(104)]).reshape(
            -1, 1
        )

        self.radius_embedding = torch.nn.Embedding.from_pretrained(
            embeddings=embeddings, freeze=True
        )
        initial_correction_embedding = torch.randn_like(embeddings) * embeddings * 0.1
        self.cov_radii_correction = torch.nn.Embedding.from_pretrained(
            embeddings=initial_correction_embedding, freeze=False
        )
        # Define the empirical parameters
        self.k0 = 7.5
        self.k1 = 4.1
        self.k2 = 19.09
        self.k3 = 254.56

    def forward(self, z, dist, edge_index):
        eps = 1e-6  # for numerical stability
        R = self.radius_embedding(z)
        Rcorrection = self.cov_radii_correction(z)
        R = R + Rcorrection
        row, col = edge_index[0], edge_index[1]
        R_i, R_j = R[row], R[col]
        Rcov_ij = R_i + R_j

        # D4
        EN = self.electronegativity_embedding(z).squeeze()
        EN_i, EN_j = EN[row], EN[col]
        delta_EN_ij = 0.5 * (
                self.k1
                * torch.exp((-1.0 / self.k3) * (torch.abs(EN_i - EN_j) + self.k2) ** 2)
        ).view(-1, 1)
        erf_arg = -self.k0 * ((dist.view(-1, 1) - Rcov_ij) / (Rcov_ij + eps))
        CN_contributions = delta_EN_ij * (1 + torch.erf(erf_arg))
        return CN_contributions


class GNN(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            num_blocks: int,
            int_emb_size: int,
            out_emb_channels: int,
            pair_basis_dim: int,
            triplet_basis_dim: int,
            output_dropout_p: float = 0.0,
            dataset_encoding_dim: int = 0,
            cutoff: float = 4.0,
            cutoff_threebody: float = 4.0
    ):
        super().__init__()

        self.cutoff = cutoff
        self.cutoff_threebody = cutoff_threebody
        assert cutoff_threebody <= cutoff
        self.num_blocks = num_blocks

        edge_feature_dim = (
                3 + dataset_encoding_dim
        )  # 2 * eeq charges, coordination edge, dataset encoding

        self.triplet_generation = Triplets(cutoff=cutoff_threebody)

        self.pair_basis = PairBasis(
            output_dim=pair_basis_dim,
            hidden_dim=3 * pair_basis_dim,
            edge_feature_dim=edge_feature_dim,
            cutoff=cutoff,
        )

        self.triplet_basis = TripletBasis(
            output_dim=triplet_basis_dim,
            hidden_dim=3 * triplet_basis_dim,
            edge_feature_dim=edge_feature_dim,
            cutoff=cutoff_threebody,
        )

        self.coordination_edges = CoordinationNumberEdges()

        self.emb = EmbeddingBlock(pair_basis_dim, hidden_channels)

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    int_emb_size=int_emb_size,
                    pair_basis_dim=pair_basis_dim,
                    triplet_basis_dim=triplet_basis_dim,
                    num_before_skip=1,
                    num_after_skip=2,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputBlock(
                    pair_basis_dim=pair_basis_dim,
                    hidden_channels=hidden_channels,
                    out_emb_channels=out_emb_channels,
                    out_channels=out_emb_channels,
                    dropout_p=output_dropout_p,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.final_readout = Linear(out_emb_channels, out_channels)

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            edge_index: Tensor,
            eeq_charges: Tensor,
            total_charge: Tensor,
            batch: Tensor,
            dataset_encoding: Optional[Tensor] = None,
    ) -> Tensor:

        vec = pos[edge_index[1]] - pos[edge_index[0]]
        dist = l2norm(vec, dim=1)

        two_body_mask = dist <= self.cutoff
        vec = vec[two_body_mask]
        dist = dist[two_body_mask]
        edge_index = edge_index[:, two_body_mask]

        idx_ji, idx_jk = self.triplet_generation.forward(vec=vec, edge_index=edge_index, n_atoms=z.shape[0])

        vec_ji, vec_jk = vec[idx_ji], vec[idx_jk]
        a = (vec_ji * vec_jk).sum(dim=-1)
        b = l2norm(torch.linalg.cross(vec_ji, vec_jk), dim=1)
        b = torch.max(b, torch.tensor(1e-9))
        angle = torch.atan2(b, a)

        coordination_edges = self.coordination_edges(
            z=z, dist=dist, edge_index=edge_index
        )

        edge_feature_list = [
            eeq_charges[edge_index[0]],
            eeq_charges[edge_index[1]],
            coordination_edges,
        ]

        if dataset_encoding is not None:
            dataset_encoding_to_edges = dataset_encoding[edge_index[0]]
            edge_feature_list.append(dataset_encoding_to_edges)

        edge_features = torch.cat(edge_feature_list, dim=1)
        pair_basis = self.pair_basis.forward(dist, edge_features)
        triplet_basis = self.triplet_basis.forward(
            dist, angle, idx_jk, idx_ji, edge_features
        )

        x = self.emb.forward(z, pair_basis, edge_index[0], edge_index[1])
        P = self.output_blocks[0].forward(x, pair_basis, edge_index[0], num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
                self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block.forward(x, pair_basis, triplet_basis, idx_jk, idx_ji)
            P = P + output_block.forward(x, pair_basis, edge_index[0], num_nodes=pos.size(0))

        out = scatter_add(P, batch, dim=0, dim_size=len(torch.unique(batch)))
        out = self.final_readout(out)
        return out
