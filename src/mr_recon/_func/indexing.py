from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    'ravel',
    'multi_index',
    'normalize',
    'multi_index_interp',
]

def ravel(x, shape, dim):
    """
    x: torch.LongTensor, arbitrary shape,
    shape: Shape of the array that x indexes into
    dim: dimension of x that is the "indexing" dimension

    Returns:
    torch.LongTensor of same shape as x but with indexing dimension removed
    """
    out = 0
    shape_shifted = tuple(shape[1:]) + (1,)
    for s, s_next, i in zip(shape, shape_shifted, range(x.shape[dim])):
        out += torch.select(x, dim, i) % s
        out *= s_next
    return out

def multi_index(x: torch.Tensor, ndims: int, idx: torch.Tensor, raveled: bool = False):
    """Extract linear indexing over last D dimensions of x, with indices given in idx
    x: [N... D...] tensor to index
    ndims: number of dimensions (from the end of the tensor) to index (length of D)
    idx: [I... ndims] or [I...] if raveled=True
    raveled: Whether the idx still needs to be raveled or not (convenient if idx is reused multiple times)

    Returns:
    Tensor with shape [N... I...] (the shape of the raveled index)
    """
    assert ndims == idx.shape[-1]
    tup = (slice(None),) * (x.ndim - ndims) + tuple(idx.moveaxis(-1, 0))
    return x[tup]
    
    x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1)
    if not raveled:
        assert ndims == idx.shape[-1], 'idx must have same last dimension as number of indexed dimensions'
        idx = ravel(idx, x.shape[-ndims:], dim=-1)
    out_shape = idx.shape
    idx_flat = torch.flatten(idx)
    y = torch.index_select(x_flat, -1, idx_flat)
    y = y.reshape((tuple(x.shape[:-ndims]) + tuple(out_shape)))
    return y

def multi_grid(x: torch.Tensor, idx: torch.Tensor, final_size: Tuple, raveled: bool = False):
    """Grid values in x to im_size with indices given in idx
    x: [N... I...]
    ndims: number of dimensions from the end of the tensor to grid (length of I)
    idx: [I... ndims] or [I...] if raveled=True
    raveled: Whether the idx still needs to be raveled or not

    Returns:
    Tensor with shape [N... final_size]

    Notes:
    Adjoint of multi_index
    """
    if not raveled:
        assert len(final_size) == idx.shape[-1], f'final_size should be of dimension {idx.shape[-1]}'
        idx = ravel(idx, final_size, dim=-1)
    ndims = len(idx.shape)
    assert x.shape[-ndims:] == idx.shape, f'x and idx should correspond in last {ndims} dimensions'
    x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1) # [N... (I...)]
    idx_flat = torch.flatten(idx)

    batch_dims = x_flat.shape[:-1]
    y = torch.zeros((*batch_dims, *final_size), dtype=x_flat.dtype, device=x_flat.device)
    y = y.reshape((*batch_dims, -1))
    y = y.index_add_(-1, idx_flat, x_flat)
    y = y.reshape(*batch_dims, *final_size)
    return y


##############
# DEPRECATED #
##############

def normalize(x: torch.Tensor, dims: Tuple):
    assert x.shape[-1] == len(dims), f'x must have {len(dims)} as the last dimension'
    dims = torch.tensor(dims, device=x.device)
    x = x / ((dims - 1) / 2) - 1.
    return x

def multi_index_interp(x: torch.Tensor, ndims: int, idx: torch.Tensor, raveled: bool = False):
    """Same as multi_index but uses grid_sample instead of index_select
    x: [N C [D] H W]
    ndims: int required for raveling
    idx: [N 1 [1] npts ndims] in xyz order i.e. idx[..., 0] indexes into the W dimension of x, idx[..., 1] indexes into H, etc.
    """
    #
    # Grid sample
    grid = normalize(idx, x.shape[-ndims:])
    y = F.grid_sample(x, grid, align_corners=True, mode='nearest')
    return y