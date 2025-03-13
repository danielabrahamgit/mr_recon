from torch import Tensor

from functools import partial
import torch

from einops import rearrange
import torch.nn.functional as F
from torch.autograd.functional import vjp


def circular_pad(t: Tensor, padding: int | tuple):
    # Infer dimension
    if isinstance(padding, int):
        ndim = t.ndim
    else:  # isinstance(padding, tuple):
        if len(padding) % 2:
            raise ValueError(
                f"Padding must have even length but got {len(padding)}: {padding}"
            )
        ndim = len(padding) // 2

    # Flatten batch dim
    batch_shape = t.shape[:-ndim]
    if ndim < len(t.shape):
        t = t.flatten(0, len(t.shape) - ndim - 1)
    else:
        # Add fake batch dim
        t = t[None]

    # Flag for cycling the batch one more time at the end
    # Avoid uselessly cycling the batch if the initial ndim is <= 3
    cycle_batch = False
    while True:
        pad_dims = min(ndim, 3)
        pad = padding[: 2 * pad_dims]
        t = circular_pad_nd(t, pad, pad_dims)

        # Update
        ndim -= pad_dims
        if ndim == 0 and not cycle_batch:
            break

        padding = padding[2 * pad_dims :]

        # Cycle remaining dims to the back, padded dims to the front
        if pad_dims == 1:
            t = rearrange(t, "... x -> x ...")
        elif pad_dims == 2:
            t = rearrange(t, "... x y  -> x y ...")
        elif pad_dims == 3:
            t = rearrange(t, "... x y z -> x y z ...")
        cycle_batch = True
        if ndim == 0:
            break

    if cycle_batch:
        # Move batch dim back to front and unflatten
        # Batch dim should be the only one left
        t = rearrange(t, "... b -> b ...")
    t = t.reshape(*batch_shape, *t.shape[1:])
    return t


def circular_pad_nd(t: Tensor, padding: int | tuple, ndim: int):
    """
    Parameters
    ----------
    t : Tensor
        The tensor to be padded
    padding : int or tuple
        Follows same conventions as torch.nn.functional.pad
    """

    # Prep
    oshape = list(t.shape)
    for i, (padleft, padright) in enumerate(zip(padding[::2], padding[1::2])):
        oshape[-(i + 1)] += padleft + padright

    if ndim == 1:
        t = rearrange(t, "... x -> (...) x")
    elif ndim == 2:
        t = rearrange(t, "... x y -> (...) x y")
    elif ndim == 3:
        t = rearrange(t, "... x y z -> (...) x y z ")
    elif ndim > 3:
        raise NotImplementedError(
            "Circular padding not yet implemented for >3 dimensions"
        )
    else:
        raise ValueError(f"ndim must be positive int but got {ndim}")

    # Do pad
    t = F.pad(t, padding, mode="circular")

    # Postproc
    t = t.reshape(*oshape)
    return t


def circular_pad_adjoint(t: Tensor, padding: int | tuple):
    """Adjoint of circular pad
    Uses vjp for compactness of implementation (may not be the most performant)
    May also not play well with autograd (hopefully this doesn't matter)
    """
    # Prep
    ishape = list(t.shape)
    for i, (padleft, padright) in enumerate(zip(padding[::2], padding[1::2])):
        ishape[-(i + 1)] -= padleft + padright

    input_ = torch.zeros(*ishape, dtype=t.dtype, device=t.device)
    f = partial(circular_pad, padding=padding)
    t = vjp(f, inputs=input_, v=t, create_graph=True, strict=False)
    return t[1]
