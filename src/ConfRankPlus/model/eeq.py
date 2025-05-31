"""
Note: Most functions of this file were hard-copied from tad-mctc==0.2.3 and tad-multicharge==0.3.3 (developed
by Marvin Friede of the Grimme Group in Bonn) and modified to be compatible with TorchScript. The changes include simple
modifications to type annotations but also more explicit changes allowing for JIT compiling.

Find the copyright statement of tad-mctc and tad-multicharge below:

SPDX-Identifier: Apache-2.0
Copyright (C) 2024 Grimme Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from __future__ import annotations
import math
import torch
from typing import Optional, List
from torch import Tensor
from tad_multicharge.param import eeq2019
from tad_mctc.data.radii import COV_D3
from tad_mctc.batch.mask.atoms import real_atoms
from tad_mctc.batch.mask.pairs import real_pairs

CUTOFF_EEQ = 25.0
CN_MAX_EEQ = 8.0
KCN_EEQ = 7.5
KCN_D3 = 16.0


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/batch/packing.py and then modified
def pack(
        tensors: List[Tensor],
        axis: int = 0,
        value: float = 0.0,
        size: Optional[List[int]] = None,
) -> Tensor:
    """
    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Parameters
    ----------
    tensors : list[Tensor]
        List of tensors to be packed, all with identical dtypes.
    axis : int
        Axis along which tensors should be packed; 0 for first axis -1
        for the last axis, etc. This will be a new dimension.
    value : float
        The value with which the tensor is to be padded.
    size : Optional[list[int]]
        Size of each dimension to which tensors should be padded.
        This to the largest size encountered along each dimension.

    Returns
    -------
    Tensor
        Input tensors padded and packed into a single tensor.
    """

    _count = len(tensors)
    _device = tensors[0].device
    _dtype = tensors[0].dtype

    # Identify the maximum size, if one was not specified.
    if size is None:
        size = [
            int(x) for x in torch.tensor([list(i.shape) for i in tensors]).max(0).values
        ]

    # Tensor to pack into, filled with padding value
    shape = [_count] + size
    padded = torch.full(shape, value, dtype=_dtype, device=_device)

    # Loop over & pack "tensors" into "padded"
    for n, source in enumerate(tensors):
        target_shape = source.shape
        if len(target_shape) == 1:
            padded[n, : target_shape[0]] = source
        elif len(target_shape) == 2:
            padded[n, : target_shape[0], : target_shape[1]] = source
        elif len(target_shape) == 3:
            padded[n, : target_shape[0], : target_shape[1], : target_shape[2]] = source
        else:
            raise Exception(
                "Packing for tensors of more than 3 dimensions not implemented yet. Try reshaping."
            )
        # Add more elif cases for higher dimensions if needed

    # If "axis" was anything other than 0, then "padded" must be permuted.
    if axis != 0:
        axis = padded.dim() + axis if axis < 0 else axis
        order = list(range(1, padded.dim()))
        order.insert(axis, 0)
        padded = padded.permute(order)

    return padded


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/storch/elemental.py and thenslightly modified
def divide(
        x: Tensor,
        y: Tensor,
        eps: float = 1e-8,
) -> Tensor:
    """
    Safe divide operation.
    Only adds a small value to the denominator where it is zero.

    Parameters
    ----------
    x : Tensor
        Input tensor (nominator).
    y : Tensor
        Input tensor (denominator).

    Returns
    -------
    Tensor
        Square root of the input tensor.

    Raises
    ------
    TypeError
        Value for addition to denominator has wrong type.
    """
    y_safe = torch.where(y == 0, torch.full_like(y, eps), y)
    return torch.divide(x, y_safe)


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/storch/distance.py and then slightly modified
def euclidean_dist_quadratic_expansion(x: Tensor, y: Tensor) -> Tensor:
    """
    Computation of euclidean distance matrix via quadratic expansion (sum of
    squared differences or L2-norm of differences).

    While this is significantly faster than the "direct expansion" or
    "broadcast" approach, it only works for euclidean (p=2) distances.
    Additionally, it has issues with numerical stability (the diagonal slightly
    deviates from zero for ``x=y``). The numerical stability should not pose
    problems, since we must remove zeros anyway for batched calculations.

    For more information, see \
    `this Jupyter notebook <https://github.com/eth-cscs/PythonHPC/blob/master/\
    numpy/03-euclidean-distance-matrix-numpy.ipynb>`__ or \
    `this discussion thread in the PyTorch forum <https://discuss.pytorch.org/\
    t/efficient-distance-matrix-computation/9065>`__.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """

    # using einsum is slightly faster than `torch.pow(x, 2).sum(-1)`
    xnorm = torch.einsum("...ij,...ij->...i", x, x)
    ynorm = torch.einsum("...ij,...ij->...i", y, y)

    n = xnorm.unsqueeze(-1) + ynorm.unsqueeze(-2)

    # "...ik,...jk->...ij"
    prod = x @ y.mT

    # important: remove negative values that give NaN in backward
    return sqrt(n - 2.0 * prod)


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/storch/distance.py and then slightly modified
def cdist_direct_expansion(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
    """
    Computation of cartesian distance matrix.

    Contrary to `euclidean_dist_quadratic_expansion`, this function allows
    arbitrary powers but is considerably slower.

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor (with same shape as first tensor).
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    # unsqueeze different dimension to create matrix
    diff = torch.abs(x.unsqueeze(-2) - y.unsqueeze(-3))

    # einsum is nearly twice as fast!
    if p == 2:
        distances = torch.einsum("...ijk,...ijk->...ij", diff, diff)
    else:
        distances = torch.sum(torch.pow(diff, p), -1)

    return torch.pow(torch.clamp(distances, min=1e-8), 1.0 / p)


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/storch/distance.py
def cdist(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
    """
    Wrapper for cartesian distance computation.

    This currently replaces the use of ``torch.cdist``, which does not handle
    zeros well and produces nan's in the backward pass.

    Additionally, ``torch.cdist`` does not return zero for distances between
    same vectors (see `here
    <https://github.com/pytorch/pytorch/issues/57690>`__).

    Parameters
    ----------
    x : Tensor
        First tensor.
    y : Tensor
        Second tensor
    p : int, optional
        Power used in the distance evaluation (p-norm). Defaults to 2.

    Returns
    -------
    Tensor
        Pair-wise distance matrix.
    """
    if y is None:
        y = x

    # faster
    if p == 2:
        return euclidean_dist_quadratic_expansion(x, y)

    return cdist_direct_expansion(x, y, p=p)


def sqrt(x: Tensor, eps: float = 1e-8):
    out = torch.sqrt(torch.clamp(x, min=eps))
    return out


def get_eps(tensor: Tensor) -> Tensor:
    if tensor.dtype == torch.float32:
        return torch.tensor(1.1920929e-07, device=tensor.device, dtype=tensor.dtype)
    elif tensor.dtype == torch.float64:
        return torch.tensor(
            2.220446049250313e-16, device=tensor.device, dtype=tensor.dtype
        )
    else:
        raise TypeError("Unsupported tensor data type")


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/ncoord/eeq.py and modified
def cut_coordination_number(cn: Tensor, cn_max: float = CN_MAX_EEQ) -> Tensor:
    if isinstance(cn_max, (float, int)):
        cn_max = torch.tensor(cn_max, device=cn.device, dtype=cn.dtype)

    if cn_max > 50:
        return cn

    return torch.log(1.0 + torch.exp(cn_max)) - torch.log(1.0 + torch.exp(cn_max - cn))


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/ncoord/count.py and modified
def erf_count(r: Tensor, r0: Tensor, kcn: float = KCN_D3) -> Tensor:
    return 0.5 * (1.0 + torch.erf(-kcn * (divide(r, r0) - 1.0)))


# copied from https://github.com/tad-mctc/tad-mctc/blob/main/src/tad_mctc/ncoord/eeq.py and modified
def cn_eeq(
        numbers: Tensor,
        positions: Tensor,
        rcov: Optional[Tensor] = None,
        cutoff: float = CUTOFF_EEQ,
        cn_max: float = CN_MAX_EEQ,
        kcn: float = KCN_EEQ,
        COV_D3: Tensor = COV_D3,
) -> Tensor:
    if cutoff is None:
        cutoff = torch.tensor(
            CUTOFF_EEQ, dtype=positions.dtype, device=positions.device
        )

    if rcov is None:
        rcov = COV_D3.to(dtype=positions.dtype, device=positions.device)
        rcov = rcov[numbers]
    else:
        rcov = rcov.to(dtype=positions.dtype, device=positions.device)

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    eps = get_eps(positions)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(mask, cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        erf_count(distances, rc, kcn),
        torch.tensor(0.0, dtype=positions.dtype, device=positions.device),
    )
    cn = torch.sum(cf, dim=-1)

    if cn_max is None:
        return cn

    return cut_coordination_number(cn, cn_max)


class EEQModel(torch.nn.Module):
    dd = {
        "device": torch.device("cuda"),
        "dtype": torch.get_default_dtype(),
    }
    """
    Electronegativity equilibration charge model published in

    - E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher,
      C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2019**, 150, 154122.
      DOI: `10.1063/1.5090222 <https://dx.doi.org/10.1063/1.5090222>`__
    """

    def __init__(
            self,
            chi: Tensor,
            kcn: Tensor,
            eta: Tensor,
            rad: Tensor,
    ) -> None:
        super().__init__()
        self.chi = torch.nn.Parameter(chi, requires_grad=False)
        self.kcn = torch.nn.Parameter(kcn, requires_grad=False)
        self.eta = torch.nn.Parameter(eta, requires_grad=False)
        self.rad = torch.nn.Parameter(rad, requires_grad=False)

    @classmethod
    def param2019(
            cls,
    ) -> EEQModel:
        """
        Create the EEQ model from the standard (2019) parametrization.

        Parameters
        ----------
        Returns
        -------
        EEQModel
            Instance of the EEQ charge model class.
        """

        return cls(
            eeq2019.chi,
            eeq2019.kcn,
            eeq2019.eta,
            eeq2019.rad,
        )

    # copied from https://github.com/tad-mctc/tad-multicharge/blob/main/src/tad_multicharge/model/eeq.py and modified
    def forward(
            self,
            numbers: Tensor,
            positions: Tensor,
            total_charge: Tensor,
            cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        eps = get_eps(positions)
        zero = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
        stop = sqrt(
            torch.tensor(2.0 / math.pi, dtype=positions.dtype, device=positions.device)
        )  # sqrt(2/pi)

        real = real_atoms(numbers)
        mask = real_pairs(numbers, mask_diagonal=True)

        distances = torch.where(
            mask,
            cdist(positions, positions, p=2),
            eps,
        )

        diagonal = mask.new_zeros(mask.shape)
        diagonal.diagonal(dim1=1, dim2=2).fill_(1.0)

        cc = torch.where(
            real,
            -self.chi[numbers] + sqrt(cn) * self.kcn[numbers],
            zero,
        )
        rhs = torch.concat((cc, total_charge.unsqueeze(1)), dim=1)

        # radii
        rad = self.rad[numbers]
        rads = rad.unsqueeze(2) ** 2 + rad.unsqueeze(1) ** 2
        gamma = torch.where(mask, 1.0 / sqrt(rads), zero)

        # hardness
        eta = torch.where(
            real,
            self.eta[numbers] + stop / rad,
            torch.tensor(1.0, dtype=positions.dtype, device=positions.device),
        )

        coulomb = torch.where(
            diagonal,
            eta.unsqueeze(2),
            torch.where(
                mask,
                torch.erf(distances * gamma) / distances,
                zero,
            ),
        )

        constraint = torch.where(
            real,
            torch.ones(numbers.shape, dtype=positions.dtype, device=positions.device),
            torch.zeros(numbers.shape, dtype=positions.dtype, device=positions.device),
        )
        zeros = torch.zeros(
            numbers.shape[:-1], dtype=positions.dtype, device=positions.device
        )

        matrix = torch.concat(
            (
                torch.concat((coulomb, constraint.unsqueeze(2)), dim=2),
                torch.concat((constraint, zeros.unsqueeze(1)), dim=1).unsqueeze(1),
            ),
            dim=1,
        )

        x = torch.linalg.solve(matrix, rhs)

        # remove constraint for energy calculation
        _x = x[..., :-1]
        _m = matrix[..., :-1, :-1]
        _rhs = rhs[..., :-1]

        # E_scalar = 0.5 * x^T @ A @ x - b @ x^T
        # E_vector =  x * (0.5 * A @ x - b)
        _e = _x * (0.5 * torch.einsum("...ij,...j->...i", _m, _x) - _rhs)
        return _e, _x
