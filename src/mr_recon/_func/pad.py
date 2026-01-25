import torch
import torch.nn as nn
import torch.nn.functional as F

class PadLast(nn.Module):
    
    def __init__(self, pad_im_size, im_size):
        super().__init__()
        assert len(pad_im_size) == len(im_size)
        self.im_dim = len(im_size)
        self.im_size = tuple(im_size)
        self.pad_im_size = tuple(pad_im_size)
        for psz in pad_im_size:
            assert not (psz % 2), 'Pad sizes must be even'

        sizes = [[(psz - isz) // 2]*2 for psz, isz in zip(pad_im_size, im_size)]
        self.pad = sum(sizes, start=[])
        self.pad.reverse()

        self.crop_slice = [slice(self.pad[2*i], -self.pad[2*i+1])
                           for i in range(len(self.pad)//2)]
        self.crop_slice.reverse()

        # Edge case
        if pad_im_size == im_size:
            self.crop_slice = [slice(None),] * self.im_dim

    def forward(self, x):
        """Pad the last n dimensions of x"""
        assert tuple(x.shape[-self.im_dim:]) == self.im_size
        pad = self.pad + [0, 0]*(x.ndim - self.im_dim)
        return F.pad(x, pad)

    def adjoint(self, y):
        """Crop the last n dimensions of y"""
        assert tuple(y.shape[-self.im_dim:]) == self.pad_im_size
        slc = [slice(None)] * (y.ndim - self.im_dim) + self.crop_slice
        return y[slc]