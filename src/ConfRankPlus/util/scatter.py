import torch
from torch import Tensor


def scatter_add(x: Tensor, idx_i: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    """
    Sum over values with the same indices.
    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce
    Returns:
        reduced input
    """
    return _scatter_add(x, idx_i, dim_size, dim)


@torch.jit.script
def _scatter_add(x: Tensor, idx_i: Tensor, dim_size: int, dim: int = 0) -> Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y
