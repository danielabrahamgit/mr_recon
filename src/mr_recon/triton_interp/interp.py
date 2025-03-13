"""Differentiable forms of Grid/Ungrid"""

from typing import Optional, Literal
from jaxtyping import Inexact, Float
from torch import Tensor
from torch.autograd import Function

import torch
from .ungrid import ungrid
from .grid import grid

__all__ = ["interpolate", "interpolate_adjoint"]


class InterpolateFn(Function):
    """Equal to block/unfold"""

    @staticmethod
    def forward(
        vals: Inexact[Tensor, "..."],
        locs: Float[Tensor, "... D"],
        width: float | tuple[float, ...],
        kernel: str,
        norm: str,
        pad_mode: str,
        kernel_params: dict,
    ) -> Tensor:
        """Remembered something about Function not liking type annotations?"""
        output = ungrid(vals, locs, width, kernel, norm, pad_mode, kernel_params)
        output.requires_grad_(vals.requires_grad)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Unpack input and output
        vals, locs, width, kernel, norm, pad_mode, kernel_params = inputs

        # Save for backward pass
        ndim = locs.shape[-1]
        ctx.grid_size = vals.shape[-ndim:]
        ctx.width = width
        ctx.kernel = kernel
        ctx.norm = norm
        ctx.pad_mode = pad_mode
        ctx.kernel_params = kernel_params
        ctx.save_for_backward(locs)

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        grad_vals = grad_locs = grad_width = grad_kernel = grad_norm = grad_pad_mode = (
            grad_kernel_params
        ) = None
        if ctx.needs_input_grad[0]:
            locs = ctx.saved_tensors[0]
            grad_vals = grid(
                grad_output,
                locs,
                ctx.grid_size,
                ctx.width,
                ctx.kernel,
                ctx.norm,
                ctx.pad_mode,
                ctx.kernel_params,
            )
        return (
            grad_vals,
            grad_locs,
            grad_width,
            grad_kernel,
            grad_norm,
            grad_pad_mode,
            grad_kernel_params,
        )


def interpolate(
    vals: Inexact[Tensor, "..."],
    locs: Float[Tensor, "... D"],
    width: float | tuple[float, ...],
    kernel="kaiser_bessel",
    norm: str = "1",
    pad_mode: str = "circular",
    kernel_params: dict = None,
):
    """Wrapper for default arguments"""
    return InterpolateFn.apply(vals, locs, width, kernel, norm, pad_mode, kernel_params)


class InterpolateAdjointFn(Function):
    """Equal to block_adjoint/fold"""

    @staticmethod
    def forward(
        vals: Inexact[Tensor, "..."],
        locs: Float[Tensor, "... D"],
        grid_size: tuple[int, ...],
        width: float | tuple[float, ...],
        kernel: str,
        norm: str,
        pad_mode: str,
        kernel_params: dict,
    ) -> Tensor:
        output = grid(
            vals,
            locs,
            grid_size,
            width,
            kernel,
            norm,
            pad_mode,
            kernel_params,
        )
        output.requires_grad_(vals.requires_grad)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        vals, locs, grid_size, width, kernel, norm, pad_mode, kernel_params = inputs

        # Save for backward pass
        ctx.width = width
        ctx.kernel = kernel
        ctx.norm = norm
        ctx.pad_mode = pad_mode
        ctx.kernel_params = kernel_params
        ctx.save_for_backward(locs)

    @staticmethod
    def backward(ctx, grad_output):
        grad_vals = grad_locs = grad_grid_size = grad_width = grad_kernel = (
            grad_norm
        ) = grad_pad_mode = grad_kernel_params = None
        if ctx.needs_input_grad[0]:
            locs = ctx.saved_tensors[0]
            grad_vals = ungrid(
                grad_output,
                locs,
                ctx.width,
                ctx.kernel,
                ctx.norm,
                ctx.pad_mode,
                ctx.kernel_params,
            )
        return (
            grad_vals,
            grad_locs,
            grad_grid_size,
            grad_width,
            grad_kernel,
            grad_norm,
            grad_pad_mode,
            grad_kernel_params,
        )


def interpolate_adjoint(
    vals: Inexact[Tensor, "..."],
    locs: Float[Tensor, "... D"],
    grid_size: tuple[int, ...],
    width: float | tuple[float, ...],
    kernel: str = "kaiser_bessel",
    norm: str = "1",
    pad_mode: str = "circular",
    kernel_params: dict = None,
):
    """Wrapper for default arguments"""
    return InterpolateAdjointFn.apply(
        vals,
        locs,
        grid_size,
        width,
        kernel,
        norm,
        pad_mode,
        kernel_params,
    )
