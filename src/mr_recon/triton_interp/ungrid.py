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
    weights4d,
    weights5d,
    weights_torch,
    get_kernel_fn,
    _apply_default_kernel_params,
    KernelTypeStr,
    mod_pos,
)
from ._batch import batch_iterator

__all__ = ["ungrid"]

# Limit maximum number of grids to launch
TRITON_MAX_GRID_SIZE = 2**16 - 1
# Limit maximum kernel width in each dimension
TRITON_MAX_KERNEL_WIDTH_1D = 2**6


def ungrid(
    vals: Inexact[Tensor, "..."],
    locs: Float[Tensor, "... D"],
    width: float | tuple[float, ...],
    kernel: str = "kaiser_bessel",
    norm: str = "1",
    pad_mode: Literal["zero", "circular"] = "circular",
    kernel_params: dict = None,
):
    """Interpolate from on-grid values to off-grid locations.

    norm: str, 1 or 2
        if 2, uses Euclidean norm to grid points to compute weights
        if 1, computes weights as product of axis-aligned norm weights
            - Same as sigpy
    """
    kernel_params = {} if kernel_params is None else kernel_params
    vals_flat, locs, shapes = prep_ungrid_shapes(vals, locs, width)
    kernel_params = _apply_default_kernel_params(kernel, kernel_params)
    out_flat = _ungrid(
        vals_flat,
        locs,
        kernel=kernel,
        norm=norm,
        pad_mode=pad_mode,
        kernel_params=kernel_params,
        **shapes,
    )
    out = out_flat.reshape(*shapes["batch_shape"], *shapes["locs_batch_shape"])
    return out


def _ungrid(
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
    if vals.is_cuda and ndim in UNGRID.keys():
        output = torch.zeros(
            nbatch,
            *locs_batch_shape,
            dtype=vals.dtype,
            device=vals.device,
        )
        if is_complex:
            vals = torch.view_as_real(vals).contiguous()
            output = torch.view_as_real(output).contiguous()
        grid = _get_grid()  # TODO
        BLOCK_WIDTH = get_block_width(width, ndim, is_complex)
        with torch.cuda.device(vals.device):
            UNGRID[ndim][grid](
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
        output = ungrid_torch(
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
def _ungrid1d(
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

    in_batch_offset = N * size
    out_batch_offset = N * npts

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

                # Load
                grid_cplx = tl.load(
                    in_ptr + in_batch_offset + x_range_cplx, x_mask_cplx
                )
                grid_mask_cplx = x_mask_cplx
                # Split and process separately
                grid_real, grid_imag = tl.split(grid_cplx)
                mask_real, mask_imag = tl.split(grid_mask_cplx)
                out_real = tl.sum(weights * grid_real * mask_real)
                out_imag = tl.sum(weights * grid_imag * mask_imag)
                tl.store(out_ptr + 2 * (out_batch_offset + p), out_real)
                tl.store(out_ptr + 2 * (out_batch_offset + p) + 1, out_imag)

            else:
                # Normal indexing
                # x_range = x_base_range + x_nbhd
                if PAD_MODE == "zero":
                    x_mask &= (x_range >= 0) & (x_range < x_size)
                elif PAD_MODE == "circular":
                    x_range = mod_pos(x_range, x_size)

                # Load
                grid = tl.load(in_ptr + in_batch_offset + x_range, x_mask)

                # Process jointly
                out = tl.sum(weights * grid * x_mask)
                tl.store(out_ptr + out_batch_offset + p, out)


@triton.heuristics(
    values={
        "pts_per_grid": lambda args: max(
            1, triton.cdiv(args["npts"] * args["nbatch"], TRITON_MAX_GRID_SIZE)
        ),
    },
)
@triton.jit
def _ungrid2d(
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

    in_batch_offset = N * size
    out_batch_offset = N * npts

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
                grid_cplx = tl.load(
                    in_ptr + in_batch_offset + grid_range_cplx, grid_mask_cplx
                )
                # Split and process separately
                grid_real, grid_imag = tl.split(grid_cplx)
                mask_real, mask_imag = tl.split(grid_mask_cplx)
                out_real = tl.sum(weights * grid_real)
                out_imag = tl.sum(weights * grid_imag)
                tl.store(out_ptr + 2 * (out_batch_offset + p), out_real)
                tl.store(out_ptr + 2 * (out_batch_offset + p) + 1, out_imag)

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
                grid = tl.load(in_ptr + in_batch_offset + grid_range, grid_mask)

                # Process jointly
                out = tl.sum(weights * grid)
                tl.store(out_ptr + out_batch_offset + p, out)


@triton.heuristics(
    values={
        "pts_per_grid": lambda args: max(
            1, triton.cdiv(args["npts"] * args["nbatch"], TRITON_MAX_GRID_SIZE)
        ),
    },
)
@triton.jit
def _ungrid3d(
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

    in_batch_offset = N * size
    out_batch_offset = N * npts

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

                # Load
                grid_cplx = tl.load(
                    in_ptr + in_batch_offset + grid_range_cplx, grid_mask_cplx
                )
                # Split and process separately
                grid_real, grid_imag = tl.split(grid_cplx)
                mask_real, mask_imag = tl.split(grid_mask_cplx)
                out_real = tl.sum(weights * grid_real * mask_real)
                out_imag = tl.sum(weights * grid_imag * mask_imag)
                tl.store(out_ptr + 2 * (out_batch_offset + p), out_real)
                tl.store(out_ptr + 2 * (out_batch_offset + p) + 1, out_imag)

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
                grid = tl.load(in_ptr + in_batch_offset + grid_range, grid_mask)

                # Process jointly
                out = tl.sum(weights * grid)
                tl.store(out_ptr + out_batch_offset + p, out)


@triton.heuristics(
    values={
        "pts_per_grid": lambda args: max(
            1, triton.cdiv(args["npts"] * args["nbatch"], TRITON_MAX_GRID_SIZE)
        ),
    },
)
@triton.jit
def _ungrid4d(
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
    w_size,
    # Size of kernel (per dimension)
    x_kernel_width,
    y_kernel_width,
    z_kernel_width,
    w_kernel_width,
    # Block sizes for each dimension
    X_BLOCK_WIDTH: tl.constexpr,
    Y_BLOCK_WIDTH: tl.constexpr,
    Z_BLOCK_WIDTH: tl.constexpr,
    W_BLOCK_WIDTH: tl.constexpr,
    pts_per_grid,  # Determined via heuristic
    beta=1.0,  # For kernel=kaiser_bessel
):
    """
    4D interpolation on a regular grid.
    For complex data, the last (w) dimension is interleaved (real, imag).
    """
    size = x_size * y_size * z_size * w_size
    if is_complex:
        # Only the last (w) dimension is interleaved,
        # so overall grid size doubles.
        size = 2 * size

    pid_0 = tl.program_id(0)
    grids_per_batch = tl.cast(tl.ceil(npts / pts_per_grid), tl.int32)
    N = pid_0 // grids_per_batch
    grid_start = pid_0 % grids_per_batch
    pts_lower = pts_per_grid * grid_start
    pts_upper = pts_per_grid * (grid_start + 1)

    # Base index ranges for each dimension
    x_base_range = tl.arange(0, X_BLOCK_WIDTH)
    y_base_range = tl.arange(0, Y_BLOCK_WIDTH)
    z_base_range = tl.arange(0, Z_BLOCK_WIDTH)
    w_base_range = tl.arange(0, W_BLOCK_WIDTH)

    in_batch_offset = N * size
    out_batch_offset = N * npts

    for p in range(pts_lower, pts_upper):
        if p < npts:
            # Load target point coordinates (4D)
            x_target = tl.load(pts_ptr + 4 * p)
            y_target = tl.load(pts_ptr + 4 * p + 1)
            z_target = tl.load(pts_ptr + 4 * p + 2)
            w_target = tl.load(pts_ptr + 4 * p + 3)

            # Compute weights and index ranges for each dimension.
            # (weights4d should return:
            #   weights, x_range, y_range, z_range, w_range,
            #   x_mask, y_mask, z_mask, w_mask)
            weights, x_range, y_range, z_range, w_range, \
                x_mask, y_mask, z_mask, w_mask = weights4d(
                    x_target, y_target, z_target, w_target,
                    x_kernel_width, y_kernel_width, z_kernel_width, w_kernel_width,
                    x_base_range, y_base_range, z_base_range, w_base_range,
                    KERNEL, NORM, beta,
                )

            if is_complex:
                # For non-last dimensions, no interleaving is required.
                x_range_cplx = x_range
                y_range_cplx = y_range
                z_range_cplx = z_range
                x_mask_cplx = x_mask
                y_mask_cplx = y_mask
                z_mask_cplx = z_mask

                # For the last dimension, interleave real and imaginary parts.
                w_range_real = 2 * w_range
                w_range_imag = 2 * w_range + 1
                w_range_cplx = tl.join(w_range_real, w_range_imag)
                w_mask_cplx = tl.join(w_mask, w_mask)

                if PAD_MODE == "zero":
                    x_mask_cplx &= (x_range_cplx >= 0) & (x_range_cplx < x_size)
                    y_mask_cplx &= (y_range_cplx >= 0) & (y_range_cplx < y_size)
                    z_mask_cplx &= (z_range_cplx >= 0) & (z_range_cplx < z_size)
                    w_mask_cplx &= (w_range_cplx >= 0) & (w_range_cplx < (2 * w_size))
                elif PAD_MODE == "circular":
                    x_range_cplx = mod_pos(x_range_cplx, x_size)
                    y_range_cplx = mod_pos(y_range_cplx, y_size)
                    z_range_cplx = mod_pos(z_range_cplx, z_size)
                    w_range_cplx = mod_pos(w_range_cplx, 2 * w_size)

                # Build the index tensor for the 4D grid.
                # For complex data, the last dimension is scaled by 2.
                grid_range_cplx = (
                    (
                        (x_range_cplx[:, None, None, None, None] * y_size +
                         y_range_cplx[None, :, None, None, None]) * z_size +
                        z_range_cplx[None, None, :, None, None]
                    ) * (w_size * 2)
                    + w_range_cplx[None, None, None, :, :]
                )
                grid_mask_cplx = (
                    x_mask_cplx[:, None, None, None, None] &
                    y_mask_cplx[None, :, None, None, None] &
                    z_mask_cplx[None, None, :, None, None] &
                    w_mask_cplx[None, None, None, :, :]
                )

                # Load the grid values (interleaved real/imag)
                grid_cplx = tl.load(in_ptr + in_batch_offset + grid_range_cplx, grid_mask_cplx)
                # Split into real and imaginary parts
                grid_real, grid_imag = tl.split(grid_cplx)
                # Here we follow the 3D example that multiplies by the masks.
                out_real = tl.sum(weights * grid_real * tl.split(grid_mask_cplx)[0])
                out_imag = tl.sum(weights * grid_imag * tl.split(grid_mask_cplx)[1])
                tl.store(out_ptr + 2 * (out_batch_offset + p), out_real)
                tl.store(out_ptr + 2 * (out_batch_offset + p) + 1, out_imag)
            else:
                # For real-valued data, adjust indices according to PAD_MODE.
                if PAD_MODE == "zero":
                    x_mask &= (x_range >= 0) & (x_range < x_size)
                    y_mask &= (y_range >= 0) & (y_range < y_size)
                    z_mask &= (z_range >= 0) & (z_range < z_size)
                    w_mask &= (w_range >= 0) & (w_range < w_size)
                elif PAD_MODE == "circular":
                    x_range = mod_pos(x_range, x_size)
                    y_range = mod_pos(y_range, y_size)
                    z_range = mod_pos(z_range, z_size)
                    w_range = mod_pos(w_range, w_size)

                grid_range = (
                    (
                        (x_range[:, None, None, None] * y_size +
                         y_range[None, :, None, None]) * z_size +
                        z_range[None, None, :, None]
                    ) * w_size
                    + w_range[None, None, None, :]
                )
                grid_mask = (
                    x_mask[:, None, None, None] &
                    y_mask[None, :, None, None] &
                    z_mask[None, None, :, None] &
                    w_mask[None, None, None, :]
                )
                grid = tl.load(in_ptr + in_batch_offset + grid_range, grid_mask)
                out = tl.sum(weights * grid)
                tl.store(out_ptr + out_batch_offset + p, out)


@triton.heuristics(
    values={
        "pts_per_grid": lambda args: max(
            1, triton.cdiv(args["npts"] * args["nbatch"], TRITON_MAX_GRID_SIZE)
        ),
    },
)
@triton.jit
def _ungrid5d(
    in_ptr,
    pts_ptr,
    out_ptr,
    nbatch,
    npts,
    KERNEL: tl.constexpr,
    NORM: tl.constexpr,
    PAD_MODE: tl.constexpr,
    is_complex: tl.constexpr,  # bool
    # Grid sizes
    x_size,
    y_size,
    z_size,
    w_size,
    v_size,
    # Kernel widths (per dimension)
    x_kernel_width,
    y_kernel_width,
    z_kernel_width,
    w_kernel_width,
    v_kernel_width,
    # Block sizes for each dimension
    X_BLOCK_WIDTH: tl.constexpr,
    Y_BLOCK_WIDTH: tl.constexpr,
    Z_BLOCK_WIDTH: tl.constexpr,
    W_BLOCK_WIDTH: tl.constexpr,
    V_BLOCK_WIDTH: tl.constexpr,
    pts_per_grid,  # Determined via heuristic
    beta=1.0,  # For kernel=kaiser_bessel
):
    """
    5D interpolation on a regular grid.
    
    For complex data, only the last (v) dimension is interleaved (real, imag),
    so the overall grid size is doubled.
    """
    # Total number of elements in the grid.
    size = x_size * y_size * z_size * w_size * v_size
    if is_complex:
        size = 2 * size  # Account for interleaved real/imag in the last dimension

    pid_0 = tl.program_id(0)
    grids_per_batch = tl.cast(tl.ceil(npts / pts_per_grid), tl.int32)
    N = pid_0 // grids_per_batch
    grid_start = pid_0 % grids_per_batch
    pts_lower = pts_per_grid * grid_start
    pts_upper = pts_per_grid * (grid_start + 1)

    # Create base index ranges for each dimension.
    x_base_range = tl.arange(0, X_BLOCK_WIDTH)
    y_base_range = tl.arange(0, Y_BLOCK_WIDTH)
    z_base_range = tl.arange(0, Z_BLOCK_WIDTH)
    w_base_range = tl.arange(0, W_BLOCK_WIDTH)
    v_base_range = tl.arange(0, V_BLOCK_WIDTH)

    in_batch_offset = N * size
    out_batch_offset = N * npts

    for p in range(pts_lower, pts_upper):
        if p < npts:
            # Load target point coordinates (5D)
            x_target = tl.load(pts_ptr + 5 * p)
            y_target = tl.load(pts_ptr + 5 * p + 1)
            z_target = tl.load(pts_ptr + 5 * p + 2)
            w_target = tl.load(pts_ptr + 5 * p + 3)
            v_target = tl.load(pts_ptr + 5 * p + 4)

            # Compute weights, index ranges, and masks for each dimension.
            # weights5d should return:
            #   weights, x_range, y_range, z_range, w_range, v_range,
            #   x_mask, y_mask, z_mask, w_mask, v_mask
            weights, x_range, y_range, z_range, w_range, v_range, \
                x_mask, y_mask, z_mask, w_mask, v_mask = weights5d(
                    x_target, y_target, z_target, w_target, v_target,
                    x_kernel_width, y_kernel_width, z_kernel_width, w_kernel_width, v_kernel_width,
                    x_base_range, y_base_range, z_base_range, w_base_range, v_base_range,
                    KERNEL, NORM, beta,
                )

            if is_complex:
                # For the first four dimensions, indices are used directly.
                x_range_cplx = x_range
                y_range_cplx = y_range
                z_range_cplx = z_range
                w_range_cplx = w_range
                x_mask_cplx = x_mask
                y_mask_cplx = y_mask
                z_mask_cplx = z_mask
                w_mask_cplx = w_mask

                # For the last (v) dimension, interleave real and imaginary parts.
                v_range_real = 2 * v_range
                v_range_imag = 2 * v_range + 1
                v_range_cplx = tl.join(v_range_real, v_range_imag)
                v_mask_cplx = tl.join(v_mask, v_mask)

                if PAD_MODE == "zero":
                    x_mask_cplx &= (x_range_cplx >= 0) & (x_range_cplx < x_size)
                    y_mask_cplx &= (y_range_cplx >= 0) & (y_range_cplx < y_size)
                    z_mask_cplx &= (z_range_cplx >= 0) & (z_range_cplx < z_size)
                    w_mask_cplx &= (w_range_cplx >= 0) & (w_range_cplx < w_size)
                    v_mask_cplx &= (v_range_cplx >= 0) & (v_range_cplx < (2 * v_size))
                elif PAD_MODE == "circular":
                    x_range_cplx = mod_pos(x_range_cplx, x_size)
                    y_range_cplx = mod_pos(y_range_cplx, y_size)
                    z_range_cplx = mod_pos(z_range_cplx, z_size)
                    w_range_cplx = mod_pos(w_range_cplx, w_size)
                    v_range_cplx = mod_pos(v_range_cplx, 2 * v_size)

                # Build the index tensor for the 5D grid.
                grid_range_cplx = (
                    (
                        (
                            (x_range_cplx[:, None, None, None, None, None] * y_size +
                             y_range_cplx[None, :, None, None, None, None]) * z_size +
                            z_range_cplx[None, None, :, None, None, None]
                        ) * w_size +
                        w_range_cplx[None, None, None, :, None, None]
                    ) * (v_size * 2) +
                    v_range_cplx[None, None, None, None, :, :]
                )
                grid_mask_cplx = (
                    x_mask_cplx[:, None, None, None, None, None] &
                    y_mask_cplx[None, :, None, None, None, None] &
                    z_mask_cplx[None, None, :, None, None, None] &
                    w_mask_cplx[None, None, None, :, None, None] &
                    v_mask_cplx[None, None, None, None, :, :]
                )

                # Load grid values (interleaved real/imag) and process.
                grid_cplx = tl.load(in_ptr + in_batch_offset + grid_range_cplx, grid_mask_cplx)
                grid_real, grid_imag = tl.split(grid_cplx)
                mask_real, mask_imag = tl.split(grid_mask_cplx)
                out_real = tl.sum(weights * grid_real * mask_real)
                out_imag = tl.sum(weights * grid_imag * mask_imag)
                tl.store(out_ptr + 2 * (out_batch_offset + p), out_real)
                tl.store(out_ptr + 2 * (out_batch_offset + p) + 1, out_imag)
            else:
                # For real-valued data, adjust indices based on PAD_MODE.
                if PAD_MODE == "zero":
                    x_mask &= (x_range >= 0) & (x_range < x_size)
                    y_mask &= (y_range >= 0) & (y_range < y_size)
                    z_mask &= (z_range >= 0) & (z_range < z_size)
                    w_mask &= (w_range >= 0) & (w_range < w_size)
                    v_mask &= (v_range >= 0) & (v_range < v_size)
                elif PAD_MODE == "circular":
                    x_range = mod_pos(x_range, x_size)
                    y_range = mod_pos(y_range, y_size)
                    z_range = mod_pos(z_range, z_size)
                    w_range = mod_pos(w_range, w_size)
                    v_range = mod_pos(v_range, v_size)

                grid_range = (
                    (
                        (
                            (x_range[:, None, None, None, None] * y_size +
                             y_range[None, :, None, None, None]) * z_size +
                            z_range[None, None, :, None, None]
                        ) * w_size +
                        w_range[None, None, None, :, None]
                    ) * v_size +
                    v_range[None, None, None, None, :]
                )
                grid_mask = (
                    x_mask[:, None, None, None, None] &
                    y_mask[None, :, None, None, None] &
                    z_mask[None, None, :, None, None] &
                    w_mask[None, None, None, :, None] &
                    v_mask[None, None, None, None, :]
                )
                grid = tl.load(in_ptr + in_batch_offset + grid_range, grid_mask)
                out = tl.sum(weights * grid)
                tl.store(out_ptr + out_batch_offset + p, out)


UNGRID = {1: _ungrid1d, 2: _ungrid2d, 3: _ungrid3d, 4: _ungrid4d, 5: _ungrid5d}


def prep_ungrid_shapes(vals, locs, width):
    ndim = locs.shape[-1]
    grid_size = tuple(vals.shape[-ndim:])

    # Flatten input vals
    batch_shape = vals.shape[:-ndim]
    if ndim < len(vals.shape):
        vals_flat = vals.flatten(0, len(vals.shape) - ndim - 1)
    else:
        vals_flat = vals[None]
    nbatch = vals_flat.shape[0]

    # Ensure locs are in [0, grid_size-1] in each dimension
    locs = torch.remainder(locs, torch.tensor(grid_size, device=locs.device))

    # Handle locs shapes
    locs_batch_shape = locs.shape[:-1]
    npts = prod(locs_batch_shape)

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


def ungrid_torch(
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
    batch size : int
        number of points to compute over at once
    """
    kernel_fn = get_kernel_fn(kernel, kernel_params)

    # Define helper vars
    nbatch = vals.shape[0]
    ndim = locs.shape[-1]
    locs_batch_shape = locs.shape[:-1]
    npts = prod(locs_batch_shape)
    device = vals.device
    grid_size = torch.tensor(grid_size, device=vals.device)

    # Get difference grid
    diff = torch.meshgrid(*(torch.arange(w + 1) for w in width), indexing="ij")
    # patch_shape = diff[0].shape
    # [prod(patch_shape), ndim]
    diff = torch.stack(diff, axis=-1).reshape(-1, ndim).to(device)
    radius = torch.tensor(width, device=device) / 2
    # [locs_batch_shape, ndim]
    locs_lower = torch.ceil(locs - radius).to(torch.int64)

    # For loop for memory purposes
    # Flatten locs batch shape
    locs = locs.reshape(-1, ndim)
    locs_lower = locs_lower.reshape(-1, ndim)
    # Create output
    out = torch.zeros(nbatch, npts, device=vals.device, dtype=vals.dtype)
    for p0, p1 in batch_iterator(npts, batch_size):
        grid_locs = locs_lower[p0:p1, None] + diff  # [prod(patch_shape), ndim]
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
        grid_locs = (slice(None), *tuple(grid_locs[..., i] for i in range(ndim)))
        out[:, p0:p1] = torch.sum(weights * vals[grid_locs] * mask, dim=-1)
    out = out.reshape(nbatch, *locs_batch_shape)
    return out
