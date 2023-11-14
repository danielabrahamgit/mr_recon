from typing import Tuple, Optional

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """Inspired by torchgeometry's extract_patches
    https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/contrib/extract_patches.html
    Works in 2D and 3D
    """
    def __init__(self, block_size: Tuple, block_stride: Tuple, input_type = None):
        """
        """
        super().__init__()
        self.input_type = input_type if input_type is not None else torch.complex64
        self.dim = len(block_size)
        self.block_size = block_size
        self.stride = block_stride

        # Hack until PyTorch supports Nd convs
        if self.dim == 2:
            self.convnd = F.conv2d
            self.convnd_t = F.conv_transpose2d
            self.unpack = unpack_blocks_2d
            self.repack = repack_blocks_2d
        elif self.dim == 3:
            self.convnd = F.conv3d
            self.convnd_t = F.conv_transpose3d
            self.unpack = unpack_blocks_3d
            self.repack = repack_blocks_3d
        else:
            raise ValueError('block dim must be 2 or 3')

        # use conv to get shape
        self.kernel = nn.Parameter(
            create_kernel(window_size=block_size, eps=0.).type(self.input_type),
            requires_grad=False,
        )

    def precompute_normalization(self, test_shape: Tuple):
        assert len(test_shape) == self.dim, 'test input must have same dim as kernel'
        test = torch.ones(
            (1, 1, *test_shape)
        ).type(self.input_type).to(self.kernel.device)
        y, nblocks = self.forward(test)
        weights = self.adjoint(y, nblocks)
        return weights

    def forward(self, x):
        """
        x: [... H W [D]]

        Returns:
        [... nblocks bH bW [bD]
        - bH, bW, bD are the dimensions of a single block
        """
        # Combine all batch/channel_dims
        batch_and_channel_dims = x.shape[:-self.dim]
        x = torch.flatten(x, start_dim=0, end_dim=-self.dim-1)[:, None, ...]

        # Conv extracts the blocks (use a special kernel)
        x = self.convnd(x, self.kernel, stride=self.stride, padding='valid')
        nblocks = x.shape[-self.dim:]
        x = self.unpack(x, self.block_size)

        # Uncombine batch/channel dims
        x = torch.unflatten(x, 0, batch_and_channel_dims)
        return x, nblocks

    def adjoint(self, x, nblocks,
                norm_weights: Optional[torch.Tensor] = None):
        # Combine batch and channel dims again
        batch_and_channel_dims = x.shape[:-len(nblocks)-1]
        x = torch.flatten(x, start_dim=0, end_dim=-len(nblocks)-2)

        # Conv transpose does the correct thing!
        x = self.repack(x, nblocks)
        x = self.convnd_t(x, self.kernel, stride=self.stride)
        # Optional normalization
        if norm_weights is not None:
            x = x / norm_weights

        # Uncombine batch/channel dims
        x = torch.unflatten(x[:, 0, ...], 0, batch_and_channel_dims)
        return x

# Helper functions
def unpack_blocks_2d(x, block_size: Tuple):
    """Move block dim from channel dim to spatial dim
    block_size = (Bh Bw)
    x: N C (Bh Bw) nBh nBw
    output:
    x: N C (nBh nBw) Bh Bw
    """
    return rearrange(x, '... (bh bw) nbh nbw -> ... (nbh nbw) bh bw',
                     bh=block_size[0])

def repack_blocks_2d(x, nblocks: Tuple):
    """
    nblocks = (nBh, nBw)
    """
    return rearrange(x, '... (nbh nbw) bh bw -> ... (bh bw) nbh nbw',
                     nbh=nblocks[0])

def unpack_blocks_3d(x, block_size: Tuple):
    """Analogous to 2D"""
    return rearrange(x, '... (bh bw bd) nbh nbw nbd -> ... (nbh nbw nbd) bh bw bd',
                     bh=block_size[0], bw=block_size[1])

def repack_blocks_3d(x, nblocks: Tuple):
    """Analogous to 3D"""
    return rearrange(x, '... (nbh nbw nbd) bh bw bd -> ... (bh bw bd) nbh nbw nbd',
                     nbh=nblocks[0], nbw=nblocks[1])

def create_kernel(
        window_size: Tuple,
        eps: float = 1e-6) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW[xD] will create a (H*W[*D])xHxW[xD] kernel.
    """
    dim = len(window_size)
    window_range: int = np.prod(window_size)
    kernel: torch.Tensor = torch.zeros((window_range, window_range)) + eps
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(*((window_range,) + (1,) + window_size))

def test_kernel():
    kernel = create_kernel(window_size=(3, 3, 3))
    breakpoint()

def test_block_select():
    x = torch.randn(1, 1, 220, 220, 220)
    window_size = (8, 8, 8)
    stride = 1

    x_blocks = block_select(x, window_size, stride)
    breakpoint()

def test_blocking_easy():
    torch.backends.cudnn.deterministic = True
    block = Block(block_size=(6, 6), block_stride=(6, 6))
    x = torch.randn(2, 2, 18, 24)
    print('x shape:', x.shape)
    y, nblocks = block(x)
    print('y shape:', y.shape)
    z = block.adjoint(y, nblocks)
    print('z shape:', z.shape)
    assert x.shape == z.shape
    assert (x == z).all().item()

def test_blocking_medium():
    torch.backends.cudnn.deterministic = True
    block = Block(block_size=(6, 6), block_stride=(3, 3))
    x = torch.randn(1, 1, 18, 24)
    y, nblocks = block(x)
    z = block.adjoint(y, nblocks)
    breakpoint()

def test_blocking_norm():
    torch.backends.cudnn.deterministic = True
    block = Block(block_size=(6, 6), block_stride=(3, 3))
    weights = block.precompute_normalization(test_shape=(18, 24))
    breakpoint()


####################################
# Worse version that uses indexing #
# ABANDONED                        #
####################################
def get_block_select(x, window_size: Tuple, stride: Tuple):
    """
    Select version is not memory efficient
    """
    N, C, *im_size = x.shape
    assert len(im_size) == len(window_size)
    dim = len(im_size)

    window_size = torch.tensor(window_size)
    im_size = torch.tensor(im_size)
    n_windows = torch.floor(im_size / window_size).long()
    window_idx = torch.stack(torch.meshgrid([torch.arange(d) for d in n_windows], indexing='ij'), dim=-1)
    window_topleft = window_idx * window_size

    subidx = torch.stack(torch.meshgrid([torch.arange(d) for d in window_size], indexing='ij'), dim=-1)

    subidx = unsqueeze_multiple(subidx, num_unsqueeze=dim, start=dim)
    idx = subidx + window_topleft
    return idx

def unsqueeze_multiple(t: torch.Tensor, num_unsqueeze: int, start: int):
    """Expand the dimensions of a tensor multiple times at some index"""
    dim = t.dim()
    assert start < dim
    slc = (slice(None),) * start + (None,) * num_unsqueeze + (slice(None),) * (dim-start)
    return t[slc]

if __name__ == '__main__':
    # test_kernel()
    # test_block_select()
    test_blocking_easy()
    #test_blocking_medium()
    test_blocking_norm()