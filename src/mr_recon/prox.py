from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Union

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sigpy as sp

from mr_recon.block import Block
from mr_recon.dtypes import complex_dtype, np_complex_dtype


__all__ = [
    'LLRHparams', 'LocallyLowRank',
]

"""
A proximal gradient is defined as 
prox_g(w) = argmin_x 1/2 ||x - w||^2 + g(x)
"""

@dataclass
class LLRHparams:
    block_size: Union[Tuple[int, int],
                      Tuple[int, int, int]]
    block_stride: Union[Tuple[int, int],
                        Tuple[int, int, int]]
    threshold: float
    rnd_shift: int = 3

def soft_thresh(x: torch.Tensor,
                rho: float) -> torch.Tensor:
    """
    Soft thresholding operator

    Parameters:
    -----------
    x : torch.Tensor
        input tensor
    rho : float
        threshold value

    Returns:
    --------
    x_thresh : torch.Tensor
        thresholded tensor
    """
    return torch.exp(1j * torch.angle(x)) * torch.max(torch.abs(x) - rho, torch.zeros(x.shape, device=x.device))

class L1Wav(nn.Module):
    """Wavelet proximal operator mimicking Sid's implimentation"""

    def __init__(self, 
                 shape: tuple, 
                 lamda: float, 
                 axes: Optional[tuple] = None,
                 rnd_shift: Optional[int] = 3,
                 wave_name: Optional[str] = 'db4'):
        """
        Parameters:
        -----------
        shape - tuple
            the image/volume dimensions
        lamda - float
            Regularization strength
        axes - tuple
            axes to compute wavelet transform over
        rnd_shift - int
            randomly shifts image by rnd_shift in each dim before applying prox
        wave_name - str
            the type of wavelet to use from:
            ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',
            'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor', ...]
            see https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#wavelet-families
        """
        super().__init__()

        # Using a fixed random number generator so that recons are consistent
        self.rng = np.random.default_rng(1000)
        self.rnd_shift = rnd_shift

        # Save wavelet params
        if axes is None:
            axes = tuple([i for i in range(len(shape))])
        self.lamda = lamda
        self.axes = axes
        self.W = sp.linop.Wavelet(shape, axes=axes, wave_name=wave_name)

    def forward(self, 
                input: torch.tensor,
                alpha: Optional[float] = 1.0):
        """
        Proximal operator for l1 wavelet

        Parameters
        ----------
        input - torch.tensor <complex> | CPU
            image/volume input
        alpha - float
            proximal 'alpha' term
        """
        
        # Random stuff
        shift = round(self.rng.uniform(-self.rnd_shift, self.rnd_shift))
        phase = np.exp(1j * self.rng.uniform(-np.pi, np.pi)).astype(np_complex_dtype)

        # Roll each axis
        nd = len(self.axes)
        input_torch = torch.roll(input, (shift,)*nd, dims=self.axes)

        # Move to sigpy
        dev = input_torch.device
        input_sigpy = input_torch.cpu().numpy()

        # Appoly random phase ...
        input_sigpy *= phase

        # Apply prox
        input_sigpy = self.W.H(sp.thresh.soft_thresh(self.lamda * alpha,
                                                        self.W(input_sigpy)))
        input_sigpy = input_sigpy.astype(np_complex_dtype)
        
        # Undo random phase ...
        input_sigpy *= np.conj(phase)
        
        # Move to pytorch
        output_torch = torch.asarray(input_sigpy).to(dev)

        # Unroll
        output_torch = torch.roll(output_torch, (-shift,)*nd, dims=self.axes)
        
        return output_torch

# FIXME TODO
class TV(nn.Module):
    
    def __init__(self,
                 im_size: tuple,
                 lamda: float,
                 norm: Optional[str] = 'l1'):
        """
        TV operator is defined as

        TV(x) = norm(D)

        Parameters:
        -----------
        im_size : tuple
            the image/volume dimensions
        lamda : float
            Regularization strength
        norm : str
            the type of norm to use from:
            ['l1', 'l2']
        """
        super().__init__()
        assert len(im_size) == 2 or len(im_size) == 3, 'Only 2D and 3D images are supported'
        self.im_size = im_size
        self.lamda = lamda
    
    def forward(self,
                input: torch.tensor,
                alpha: Optional[float] = 1.0):
        """
        Proximal operator

        Parameters:
        -----------
        input : torch.tensor
            image/volume input with shape (..., *im_size)
        alpha : float
            proximal weighting term on g(x)

        Returns:
        --------
        output : torch.tensor
            proximal output
        """

        return input # TODO

class LocallyLowRank(nn.Module):
    """Version of LLR mimicking Sid's version in Sigpy

    Language based on spatiotemporal blocks
    """
    def __init__(
            self,
            input_size: Tuple,
            hparams: LLRHparams,
            input_type: Optional[Callable]= None,
    ):
        super().__init__()
        self.input_type = input_type if input_type is not None else complex_dtype
        self.hparams = hparams

        # Using a fixed random number generator so that recons are consistent
        self.rng = np.random.default_rng(1000)
        self.rnd_shift = hparams.rnd_shift

        # Derived
        self.block = Block(self.hparams.block_size, self.hparams.block_stride)
        self.block_weights = nn.Parameter(
            self.block.precompute_normalization(input_size).type(self.input_type),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [N A H W [D]]
          - N: Batch dim
          - A: Temporal (subspace) dim
          - H, W, [D]: spatial dims

        """
        assert x.dim() >= 4
        block_dim = len(self.block.block_size)

        # Random shift 
        shift = round(self.rng.uniform(-self.rnd_shift, self.rnd_shift))

        # Roll in each axis by some shift amount
        x = torch.roll(x, (shift,)*block_dim, dims=tuple(range(-block_dim, 0)))

        # Extract Blocks
        x, nblocks = self.block(x)

        # Combine within-block dimensions
        # Move temporal dimension to be second-to-last
        unblocked_shape = x.shape # Save block shape for later
        x = rearrange(x, 'n a b ... -> n b a (...)')

        # Take SVD
        U, S, Vh = torch.linalg.svd(x, full_matrices=False, driver='gesvda')

        # Threshold
        S = S - self.hparams.threshold
        S[S < 0] = 0.
        S = S.type(U.dtype)

        # Recompose blocks
        x = U @ (S[..., None] * Vh)

        # Unblock and normalize
        x = rearrange(x, 'n b a x -> n a b x')
        x = x.reshape(*unblocked_shape)
        x = self.block.adjoint(x, nblocks, norm_weights=self.block_weights)

        # Undo the roll in each shift direction
        x = torch.roll(x, (-shift,)*block_dim, dims=tuple(range(-block_dim, 0)))

        # Return the thresholded input
        return x

    def forward_mrf(self, x: torch.Tensor):
        """Simple wrapper that fixes dimensions
        x: [A H W [D]]

        Adds batch dim
        """
        assert x.dim() == 3 or x.dim() == 4
        x = x[None, ...]
        x = self(x)
        x = x[0, ...]
        return x