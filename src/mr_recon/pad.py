import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Literal
class PadLast(nn.Module):
    """
    Zero-pad the last `im_dim` dimensions from `im_size` to `pad_im_size`.
    Supports odd differences by assigning the extra 1 to either the left or right side.

    Parameters
    ----------
    pad_im_size : Sequence[int]
        Target spatial size (last dims).
    im_size : Sequence[int]
        Original spatial size (last dims).
    """
    def __init__(
        self,
        pad_im_size: Sequence[int],
        im_size: Sequence[int],
    ):
        super().__init__()
        assert len(pad_im_size) == len(im_size)
        self.im_dim = len(im_size)
        self.im_size = tuple(im_size)
        self.pad_im_size = tuple(pad_im_size)

        # Require padding (no negative diffs here)
        diffs = [P - I for P, I in zip(self.pad_im_size, self.im_size)]
        if any(d < 0 for d in diffs):
            raise ValueError("pad_im_size must be >= im_size in every dim")
        
        pads_lr = []
        for (P, I) in zip(self.pad_im_size, self.im_size):
            delta = P - I
            left = delta // 2 # floor
            right = delta - left # ceil
            pads_lr.append((left, right))
        self.pads_lr = tuple(pads_lr)

        pad_list = []
        for l, r in reversed(self.pads_lr):
            pad_list.extend([l, r])
        self.pad = pad_list 

        self.crop_slice = [slice(l, l + I) for (l, _), I in zip(self.pads_lr, self.im_size)]

        # edge case
        if all(d == 0 for d in diffs):
            self.crop_slice = [slice(None)] * self.im_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the last n dimensions of x"""
        assert tuple(x.shape[-self.im_dim:]) == self.im_size
        pad = self.pad + [0, 0] * (x.ndim - self.im_dim)  # keep leading dims unchanged
        return F.pad(x, pad)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Crop the last n dimensions of y"""
        assert tuple(y.shape[-self.im_dim:]) == self.pad_im_size
        slc = [slice(None)] * (y.ndim - self.im_dim) + self.crop_slice
        return y[tuple(slc)]