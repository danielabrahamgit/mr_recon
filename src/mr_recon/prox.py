from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Union

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sigpy as sp
import ptwt

from mr_recon.block import Block
from mr_recon.dtypes import complex_dtype, np_complex_dtype

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
    """Wavelet proximal operator"""

    def __init__(self, 
                 im_size: tuple, 
                 lamda: float, 
                 axes: Optional[tuple] = None,
                 rnd_shift: Optional[int] = 3,
                 level: Optional[int] = None,
                 rnd_phase: Optional[bool] = True,
                 wave_name: Optional[str] = 'db4'):
        """
        Parameters:
        -----------
        shape - tuple
            the image/volume dimensions
        lamda - float
            Regularization strength
        level - int
            the level of wavelet decomposition
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

        # Default over last len(im_size) axes
        if axes is None:
            axes = tuple([d for d in range(-len(im_size), 0)])
        self.im_size = im_size
        self.lamda = lamda
        self.axes = axes
        self.rnd_shift = rnd_shift
        self.rnd_phase = rnd_phase
        w = ptwt.fswavedec2 if len(im_size) == 2 else ptwt.wavedec3
        wh = ptwt.fswaverec2 if len(im_size) == 2 else ptwt.waverec3
        def dec(x):
            # Stack real and imaginary parts
            x_stack = torch.cat((x.real, x.imag), dim=0)
            return w(x_stack, wave_name, axes=axes, level=level)
        def rec(x):
            # Inverse stack real and imaginary parts
            stacked = wh(x, wave_name, axes=axes)
            re_im = stacked[:len(stacked)//2] + 1j * stacked[len(stacked)//2:]
            return re_im.type(complex_dtype)
        self.dec = dec
        self.rec = rec
        
    @staticmethod
    def soft_threshold_coeffs_re_im(coeffs, tau):
        # Real is fist half, imaginary is second half
        # helper for a single tensor
        def _soft_(x):
            # sign(x) * max(|x| - tau, 0)
            # we do it in‐place on x
            x_real = x[:len(x)//2]
            x_imag = x[len(x)//2:]
            mag = torch.sqrt(x_real**2 + x_imag**2)
            phz = torch.atan2(x_imag, x_real)
            mag = torch.max(mag - tau, torch.zeros_like(mag))
            x[:len(x)//2] = mag * torch.cos(phz)
            x[len(x)//2:] = mag * torch.sin(phz)
            return x
        
        # Apply soft thresholding to each level
        _soft_(coeffs[0])
        for level_dict in coeffs[1:]:
            for key, arr in level_dict.items():
                _soft_(arr)
        return coeffs


    def forward(self, 
                input: torch.tensor,
                alpha: Optional[float] = 1.0):
        """
        Proximal operator for l1 wavelet

        Parameters
        ----------
        input - torch.tensor
            image/volume input with shape (N, *im_size)
        alpha - float
            proximal 'alpha' term
        """
        
        if input.dim() == len(self.im_size):
            # Add batch dim
            input = input[None, ...]
            batch = False
        elif input.dim() != len(self.im_size) + 1:
            raise ValueError(f'Input must have {len(self.im_size) + 1} dimensions, got {input.dim()}')
        else:
            batch = False
        assert input.shape[-len(self.im_size):] == self.im_size, \
            f'Input shape {input.shape} does not match im_size {self.im_size}'
        
        # Random stuff
        shifts = torch.randint(-self.rnd_shift,
                               self.rnd_shift + 1, 
                               (len(self.im_size),)).tolist()
        shifts_neg = [-s for s in shifts]
        if self.rnd_phase:
            phz = torch.rand((1,), device=input.device) * 2 * np.pi 
            phase = torch.exp(1j * phz).type(complex_dtype)
        else:
            phase = torch.tensor([1.0], dtype=complex_dtype, device=input.device)

        # Apply Randomness
        nd = len(self.axes)
        input_shift = input.roll(shifts, dims=self.axes) * phase

        # Apply prox
        coeffs = self.dec(input_shift)
        coeffs_st = self.soft_threshold_coeffs_re_im(coeffs, alpha * self.lamda)
        input_st = self.rec(coeffs_st)
        
        # Undo random stuff ...
        output = (input_st * phase.conj()).roll(shifts_neg, dims=self.axes)
        
        if batch:
            return output
        return output[0, ...]

class L1Wav_cpu(nn.Module):
    """Wavelet proximal operator mimicking Sid's implimentation"""

    def __init__(self, 
                 shape: tuple, 
                 lamda: float, 
                 axes: Optional[tuple] = None,
                 rnd_shift: Optional[int] = 3,
                 rnd_phase: Optional[bool] = True,
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
        rnd_phase - bool
            whether to apply a random phase to the image before applying prox
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
        self.rnd_phase = rnd_phase
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
        if self.rnd_phase:
            phase = np.exp(1j * self.rng.uniform(-np.pi, np.pi)).astype(np_complex_dtype)
        else:
            phase = 1.0

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
        Vh.nan_to_num_(0.0)

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