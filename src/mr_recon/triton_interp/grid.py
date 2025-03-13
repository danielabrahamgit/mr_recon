from typing import Literal
from jaxtyping import Shaped, Float, Inexact
from torch import Tensor

from functools import partial
from math import prod, ceil

import torch
import triton
import triton.language as tl

from .kernels import (
    weights1d,
    weights2d,
    weights3d,
    weights_torch,
    get_kernel_fn,
    _apply_default_kernel_params,
    KernelTypeStr,
    mod_pos,
)
from ._batch import batch_iterator

__all__ = ["grid"]

# Limit maximum number of grids to launch
TRITON_MAX_GRID_SIZE = 2**16 - 1
# Limit maximum kernel width in each dimension
TRITON_MAX_KERNEL_WIDTH_1D = 2**6


def grid(
    vals: Inexact[Tensor, "..."],
    locs: Float[Tensor, "... D"],
    grid_size: tuple[int, ...],
    width: float | tuple[float, ...],
    kernel: str = "kaiser_bessel",
    norm: str = "1",
    pad_mode: Literal["zero", "circular"] = "circular",
    kernel_params: dict = None,
):
    """Interpolate from off-grid values to on-grid locations.

    grid_size : tuple[int, ...]
        Shape of output array, excluding batch dimensions
        For example, if the gridded output should have shape [3, 64, 64],
        where 3 is the batch size, then grid_size would be [64, 64]

    norm: str, 1 or 2
        if 2, uses Euclidean norm to grid points to compute weights
        if 1, computes weights as product of axis-aligned norm weights
            - Same as sigpy
    """
    kernel_params = {} if kernel_params is None else kernel_params
    vals_flat, locs, shapes = prep_grid_shapes(vals, locs, grid_size, width)
    kernel_params = _apply_default_kernel_params(kernel, kernel_params)
    out_flat = _grid(
        vals_flat,
        locs,
        kernel=kernel,
        norm=norm,
        pad_mode=pad_mode,
        kernel_params=kernel_params,
        **shapes,
    )
    out = out_flat.reshape(*shapes["batch_shape"], *grid_size)
    return out


def _grid(
    vals: Shaped[Tensor, "B ..."],
    locs: Float[Tensor, "... D"],
    grid_size: tuple[int, ...],
    width: tuple[float, ...],
    kernel: KernelTypeStr,
    norm: str,
    pad_mode: str,
    ndim: int,
    nbatch: int,
    is_complex: bool,
    locs_batch_shape: tuple[int, ...],
    npts: int,
    kernel_params,
    **kwargs,
):
    if vals.is_cuda and ndim in GRID.keys():
        output = torch.zeros(
            nbatch,
            *grid_size,
            dtype=vals.dtype,
            device=vals.device,
        )
        if is_complex:
            vals = torch.view_as_real(vals).contiguous()
            output = torch.view_as_real(output).contiguous()
        grid = _get_grid()
        BLOCK_WIDTH = get_block_width(width, ndim, is_complex)
        with torch.cuda.device(vals.device):
            GRID[ndim][grid](
                vals,
                locs,
                output,
                nbatch,
                npts,
                kernel,
                norm,
                pad_mode,
                is_complex,
                *grid_size,
                *width,
                *BLOCK_WIDTH,
                **kernel_params,
            )
        if is_complex:
            output = torch.view_as_complex(output)
    else:
        # Pytorch fallback
        output = grid_torch(
            vals,
            locs,
            grid_size,
            width,
            kernel,
            norm,
            pad_mode,
            **kernel_params,
        )
    return output


def _get_grid():
    grid = lambda meta: (ceil(meta["npts"] / meta["pts_per_grid"]) * meta["nbatch"],)  # noqa: E731
    return grid


def get_block_width(kernel_width: tuple[float, ...], ndim: int, is_complex: bool):
    """Get necessary block width based on dimension and dtype of input"""
    block_width = list(triton.next_power_of_2(ceil(w + 1)) for w in kernel_width)
    test_block_width = block_width[:]  # Shallow copy
    if is_complex:
        test_block_width[-1] *= 2
    if max(test_block_width) > TRITON_MAX_KERNEL_WIDTH_1D:
        raise ValueError(
            f"Necessary block width {test_block_width} has entry which exceeds maximum width {TRITON_MAX_KERNEL_WIDTH_1D} (kernel width is doubled in last dim if input is complex)"
        )

    return block_width


@triton.heuristics(
    values={
        "pts_per_grid": lambda args: max(
            1, triton.cdiv(args["npts"] * args["nbatch"], TRITON_MAX_GRID_SIZE)
        ),
    },
)
@triton.jit
def _grid1d(
    in_ptr,
    pts_ptr,
    out_ptr,
    nbatch,
    npts,
    KERNEL: tl.constexpr,
    NORM: tl.constexpr,
    PAD_MODE: tl.constexpr,
    is_complex: tl.constexpr,  # bool
    x_size,
    x_kernel_width,
    X_BLOCK_WIDTH: tl.constexpr,
    pts_per_grid,  # Determined via heuristic
    beta=1.0,  # For kernel=kaiser_bessel
):
    """
    NORM has no effect in 1d
    """
    size = x_size

    pid_0 = tl.program_id(0)
    grids_per_batch = tl.cast(tl.ceil(npts / pts_per_grid), tl.int32)
    N, grid_start = pid_0 // grids_per_batch, pid_0 % grids_per_batch
    pts_lower, pts_upper = pts_per_grid * grid_start, pts_per_grid * (grid_start + 1)

    x_base_range = tl.arange(0, X_BLOCK_WIDTH)
    if is_complex:
        # Last dimension has double size because (real, imag) are interleaved
        # Overall size doubles as a result
        size = 2 * size

    in_batch_offset = N * npts
    out_batch_offset = N * size

    for p in range(pts_lower, pts_upper):
        if p < npts:
            x_target = tl.load(pts_ptr + p)
            weights, x_range, x_mask = weights1d(
                x_target, x_kernel_width, x_base_range, KERNEL, beta
            )

            if is_complex:
                # Pytorch interleaved indexing
                # Only applies to last dimension
                x_range_real = 2 * x_range
                x_range_imag = 2 * x_range + 1
                x_range_cplx = tl.join(x_range_real, x_range_imag)  # [width, 2]
                x_mask_cplx = tl.join(x_mask, x_mask)
                if PAD_MODE == "zero":
                    x_mask_cplx &= (x_range_cplx >= 0) & (x_range_cplx < (2 * x_size))
                elif PAD_MODE == "circular":
                    x_range_cplx = mod_pos(x_range_cplx, 2 * x_size)

                # Split and process separately
                pt_real = tl.load(in_ptr + 2 * (in_batch_offset + p))
                pt_imag = tl.load(in_ptr + 2 * (in_batch_offset + p) + 1)
                out_real = weights * pt_real
                out_imag = weights * pt_imag
                out_cplx = tl.join(out_real, out_imag)

                # Store
                tl.atomic_add(
                    out_ptr + out_batch_offset + x_range_cplx, out_cplx, x_mask_cplx
                )

            else:
                # Normal indexing
                if PAD_MODE == "zero":
                    x_mask &= (x_range >= 0) & (x_range < x_size)
                elif PAD_MODE == "circular":
                    x_range = mod_pos(x_range, x_size)

                # Load
                pt_val = tl.load(in_ptr + in_batch_offset + p)
                out = weights * pt_val

                # Store
                tl.atomic_add(out_ptr + out_batch_offset + x_range, out, x_mask)


@triton.heuristics(
    values={
        "pts_per_grid": lambda args: max(
            1, triton.cdiv(args["npts"] * args["nbatch"], TRITON_MAX_GRID_SIZE)
        ),
    },
)
@triton.jit
def _grid2d(
    in_ptr,
    pts_ptr,
    out_ptr,
    nbatch,
    npts,
    KERNEL: tl.constexpr,
    NORM: tl.constexpr,
    PAD_MODE: tl.constexpr,
    is_complex: tl.constexpr,  # bool
    # Size of grid
    x_size,
    y_size,
    # Size of kernel
    x_kernel_width,
    y_kernel_width,
    # Size of blocks to load
    X_BLOCK_WIDTH: tl.constexpr,
    Y_BLOCK_WIDTH: tl.constexpr,
    pts_per_grid,  # Determined via heuristic
    beta=1.0,  # For kernel=kaiser_bessel
):
    """ """
    size = x_size * y_size

    pid_0 = tl.program_id(0)
    grids_per_batch = tl.cast(tl.ceil(npts / pts_per_grid), tl.int32)
    N, grid_start = pid_0 // grids_per_batch, pid_0 % grids_per_batch
    pts_lower, pts_upper = pts_per_grid * grid_start, pts_per_grid * (grid_start + 1)

    x_base_range = tl.arange(0, X_BLOCK_WIDTH)
    y_base_range = tl.arange(0, Y_BLOCK_WIDTH)
    if is_complex:
        # Last dimension has double size because (real, imag) are interleaved
        # Overall size doubles as a result
        size = 2 * size

    in_batch_offset = N * npts
    out_batch_offset = N * size

    for p in range(pts_lower, pts_upper):
        if p < npts:
            # Load target point
            x_target = tl.load(pts_ptr + 2 * p)
            y_target = tl.load(pts_ptr + 2 * p + 1)

            weights, x_range, y_range, x_mask, y_mask = weights2d(
                x_target,
                y_target,
                x_kernel_width,
                y_kernel_width,
                x_base_range,
                y_base_range,
                KERNEL,
                NORM,
                beta,
            )

            if is_complex:
                x_range_cplx = x_range  # [width]
                x_mask_cplx = x_mask
                # Pytorch interleaved indexing
                # Only applies to last dimension
                y_range_real = 2 * y_range
                y_range_imag = 2 * y_range + 1
                y_range_cplx = tl.join(y_range_real, y_range_imag)  # [width, 2]
                y_mask_cplx = tl.join(y_mask, y_mask)
                if PAD_MODE == "zero":
                    x_mask_cplx &= (x_range_cplx >= 0) & (x_range_cplx < x_size)
                    y_mask_cplx &= (y_range_cplx >= 0) & (y_range_cplx < (2 * y_size))
                elif PAD_MODE == "circular":
                    x_range_cplx = mod_pos(x_range_cplx, x_size)
                    y_range_cplx = mod_pos(y_range_cplx, 2 * y_size)

                grid_range_cplx = (
                    x_range_cplx[:, None, None] * y_size * 2 + y_range_cplx[None, :, :]
                )
                grid_mask_cplx = x_mask_cplx[:, None, None] & y_mask_cplx[None, :, :]

                # Load
                # Split and process separately
                pt_real = tl.load(in_ptr + 2 * (in_batch_offset + p))
                pt_imag = tl.load(in_ptr + 2 * (in_batch_offset + p) + 1)
                out_real = weights * pt_real
                out_imag = weights * pt_imag
                out_cplx = tl.join(out_real, out_imag)

                # Accumulate
                tl.atomic_add(
                    out_ptr + out_batch_offset + grid_range_cplx,
                    out_cplx,
                    grid_mask_cplx,
                )

            else:
                # Normal indexing
                if PAD_MODE == "zero":
                    x_mask &= (x_range >= 0) & (x_range < x_size)
                    y_mask &= (y_range >= 0) & (y_range < y_size)
                elif PAD_MODE == "circular":
                    x_range = mod_pos(x_range, x_size)
                    y_range = mod_pos(y_range, y_size)

                grid_range = x_range[:, None] * y_size + y_range[None, :]
                grid_mask = x_mask[:, None] & y_mask[None, :]

                # Load
                pt_val = tl.load(in_ptr + in_batch_offset + p)
                out = weights * pt_val

                # Accumulate
                tl.atomic_add(out_ptr + out_batch_offset + grid_range, out, grid_mask)


@triton.heuristics(
    values={
        "pts_per_grid": lambda args: max(
            1, triton.cdiv(args["npts"] * args["nbatch"], TRITON_MAX_GRID_SIZE)
        ),
    },
)
@triton.jit
def _grid3d(
    in_ptr,
    pts_ptr,
    out_ptr,
    nbatch,
    npts,
    KERNEL: tl.constexpr,
    NORM: tl.constexpr,
    PAD_MODE: tl.constexpr,
    is_complex: tl.constexpr,  # bool
    # Size of grid
    x_size,
    y_size,
    z_size,
    # Size of kernel
    x_kernel_width,
    y_kernel_width,
    z_kernel_width,
    # Size of blocks to load
    X_BLOCK_WIDTH: tl.constexpr,
    Y_BLOCK_WIDTH: tl.constexpr,
    Z_BLOCK_WIDTH: tl.constexpr,
    pts_per_grid,  # Determined via heuristic
    beta=1.0,  # For kernel=kaiser_bessel
):
    """ """
    size = x_size * y_size * z_size

    pid_0 = tl.program_id(0)
    grids_per_batch = tl.cast(tl.ceil(npts / pts_per_grid), tl.int32)
    N, grid_start = pid_0 // grids_per_batch, pid_0 % grids_per_batch
    pts_lower, pts_upper = pts_per_grid * grid_start, pts_per_grid * (grid_start + 1)

    x_base_range = tl.arange(0, X_BLOCK_WIDTH)
    y_base_range = tl.arange(0, Y_BLOCK_WIDTH)
    z_base_range = tl.arange(0, Z_BLOCK_WIDTH)
    if is_complex:
        # Last dimension has double size because (real, imag) are interleaved
        # Overall size doubles as a result
        size = 2 * size

    in_batch_offset = N * npts
    out_batch_offset = N * size

    for p in range(pts_lower, pts_upper):
        if p < npts:
            # Load target point
            x_target = tl.load(pts_ptr + 3 * p)
            y_target = tl.load(pts_ptr + 3 * p + 1)
            z_target = tl.load(pts_ptr + 3 * p + 2)

            weights, x_range, y_range, z_range, x_mask, y_mask, z_mask = weights3d(
                x_target,
                y_target,
                z_target,
                x_kernel_width,
                y_kernel_width,
                z_kernel_width,
                x_base_range,
                y_base_range,
                z_base_range,
                KERNEL,
                NORM,
                beta,
            )

            if is_complex:
                x_range_cplx = x_range  # [width]
                y_range_cplx = y_range
                x_mask_cplx = x_mask
                y_mask_cplx = y_mask
                # Pytorch interleaved indexing
                # Only applies to last dimension
                z_range_real = 2 * z_range  # 2 is for real/complex, not dimension
                z_range_imag = 2 * z_range + 1
                z_range_cplx = tl.join(z_range_real, z_range_imag)  # [width, 2]
                z_mask_cplx = tl.join(z_mask, z_mask)
                if PAD_MODE == "zero":
                    x_mask_cplx &= (x_range_cplx >= 0) & (x_range_cplx < x_size)
                    y_mask_cplx &= (y_range_cplx >= 0) & (y_range_cplx < y_size)
                    z_mask_cplx &= (z_range_cplx >= 0) & (z_range_cplx < (2 * z_size))
                elif PAD_MODE == "circular":
                    x_range_cplx = mod_pos(x_range_cplx, x_size)
                    y_range_cplx = mod_pos(y_range_cplx, y_size)
                    z_range_cplx = mod_pos(z_range_cplx, 2 * z_size)

                grid_range_cplx = (
                    x_range_cplx[:, None, None, None] * y_size
                    + y_range_cplx[None, :, None, None]
                ) * z_size * 2 + z_range_cplx[None, None, :, :]
                grid_mask_cplx = (
                    x_mask_cplx[:, None, None, None] & y_mask_cplx[None, :, None, None]
                ) & z_mask_cplx[None, None, :, :]

                # Load split
                pt_real = tl.load(in_ptr + 2 * (in_batch_offset + p))
                pt_imag = tl.load(in_ptr + 2 * (in_batch_offset + p) + 1)
                out_real = weights * pt_real
                out_imag = weights * pt_imag
                out_cplx = tl.join(out_real, out_imag)

                # Accmulate
                tl.atomic_add(
                    out_ptr + out_batch_offset + grid_range_cplx,
                    out_cplx,
                    grid_mask_cplx,
                )

            else:
                # Normal indexing
                if PAD_MODE == "zero":
                    x_mask &= (x_range >= 0) & (x_range < x_size)
                    y_mask &= (y_range >= 0) & (y_range < y_size)
                    z_mask &= (z_range >= 0) & (z_range < z_size)
                elif PAD_MODE == "circular":
                    x_range = mod_pos(x_range, x_size)
                    y_range = mod_pos(y_range, y_size)
                    z_range = mod_pos(z_range, z_size)

                grid_range = (
                    x_range[:, None, None] * y_size + y_range[None, :, None]
                ) * z_size + z_range[None, None, :]
                grid_mask = (
                    x_mask[:, None, None]
                    & y_mask[None, :, None]
                    & z_mask[None, None, :]
                )

                # Load
                pt_val = tl.load(in_ptr + in_batch_offset + p)
                out = weights * pt_val

                # Accumulate
                tl.atomic_add(out_ptr + out_batch_offset + grid_range, out, grid_mask)


GRID = {1: _grid1d, 2: _grid2d, 3: _grid3d}


def prep_grid_shapes(vals, locs, grid_size, width):
    ndim = len(grid_size)
    if locs.shape[-1] != ndim:
        raise ValueError(
            f"Expected last dim of locs: to equal ndim but got locs.shape[-1] {locs.shape[-1]} != ndim {ndim}"
        )
    # Handle locs shapes
    locs_batch_shape = locs.shape[:-1]
    npts = prod(locs_batch_shape)

    # Ensure locs are in [0, grid_size-1] in each dimension
    locs = torch.remainder(locs, torch.tensor(grid_size, device=locs.device))

    # Flatten input vals
    batch_shape = vals.shape[: -len(locs_batch_shape)]
    if len(locs_batch_shape) < len(vals.shape):
        vals_flat = vals.flatten(0, len(vals.shape) - len(locs_batch_shape) - 1)
    else:
        vals_flat = vals[None]
    nbatch = vals_flat.shape[0]

    # Complex input
    is_complex = torch.is_complex(vals)

    # Ensure kernel width is a tuple
    if isinstance(width, float):
        width = (width,) * ndim
    elif isinstance(width, tuple) and len(width) != ndim:
        raise ValueError(
            f"If width specified as tuple it must be same length as grid size but got len(width) = {len(width)} and len(grid_size) = {ndim}"
        )

    return (
        vals_flat,
        locs,
        {
            "ndim": ndim,
            "grid_size": grid_size,
            "width": width,
            "nbatch": nbatch,
            "batch_shape": batch_shape,
            "is_complex": is_complex,
            "locs_batch_shape": locs_batch_shape,
            "npts": npts,
        },
    )


def grid_torch(
    vals: Inexact[Tensor, "B ..."],
    locs: Float[Tensor, "... D"],
    grid_size: tuple[int, ...],
    width: tuple[float, ...],
    kernel: str = "kaiser_bessel",
    norm: str = "1",
    pad_mode: Literal["zero", "circular"] = "circular",
    batch_size: int = 2**20,
    **kernel_params,
):
    """Torch fallback

    Eventually, may want to use triton's CPU backend

    pad_mode : 'zero' or 'circular'
        Type of edge behavior to use
    batch_size : int
        number of points to compute over at once
    """

    kernel_fn = get_kernel_fn(kernel, kernel_params)

    # Define helper vars
    ndim = locs.shape[-1]
    device = vals.device
    nbatch = vals.shape[0]
    diff = torch.meshgrid(*(torch.arange(w + 1) for w in width), indexing="ij")
    # patch_size = diff[0].shape
    grid_size = torch.tensor(grid_size, device=vals.device)

    # [prod(patch_size), ndim]
    diff = torch.stack(diff, axis=-1).reshape(-1, ndim).to(device)
    radius = torch.tensor(width, device=device) / 2
    locs_lower = torch.ceil(locs - radius).to(torch.int64)

    # For loop for memory purposes
    out = torch.zeros(nbatch, *tuple(grid_size), dtype=vals.dtype, device=device)
    vals = vals.reshape(nbatch, -1)  # [nbatch, npts]
    npts = vals.shape[1]
    locs = locs.reshape(-1, ndim)  # [npts, ndim]
    locs_lower = locs_lower.reshape(-1, ndim)  # [npts, ndim]
    # Prepare batch indices (memory efficient because of expand())
    batch_indices = torch.arange(nbatch).view(-1, 1, 1)  # [B npts ngrid]
    for p0, p1 in batch_iterator(npts, batch_size):
        # [npts, npatch, ndim]
        grid_locs = locs_lower[p0:p1, None] + diff
        # Normalized delta(locations)
        weights, grid_locs, mask = weights_torch(
            locs[p0:p1],
            grid_locs,
            radius,
            norm,
            kernel_fn,
            grid_size,
            pad_mode,
        )
        val = vals[:, p0:p1, None]  # [nbatch, npts, 1]
        patches = weights * mask * val

        # Expand patch indices
        npts = p1 - p0
        ngrid = patches.shape[-1]
        batch_indices = batch_indices.expand(-1, npts, ngrid)

        # Perform indexing
        grid_locs_slc = (batch_indices, *(grid_locs[..., i] for i in range(ndim)))
        out.index_put_(grid_locs_slc, patches, accumulate=True)
    return out
