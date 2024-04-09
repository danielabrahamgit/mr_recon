import torch
import torch.nn as nn
import numpy as np
import sigpy as sp
import torch.fft as fft_torch

from typing import Optional
from torchkbnufft import KbNufft, KbNufftAdjoint
from mr_recon.pad import PadLast
from mr_recon.indexing import (
    multi_grid,
    multi_index
)
from sigpy.fourier import (
    _get_oversamp_shape, 
    _apodize, 
    _scale_coord)
from mr_recon.utils import (
    torch_to_np, 
    np_to_torch,
    batch_iterator)

def fft(x, dim=None, oshape=None, norm='ortho'):
    """Matches Sigpy's fft, but in torch"""

    if oshape is not None:
        x_cp = torch_to_np(x)
        dev = sp.get_device(x_cp)
        with dev:
            x = np_to_torch(sp.resize(x_cp, oshape))
    x = fft_torch.ifftshift(x, dim=dim)
    x = fft_torch.fftn(x, dim=dim, norm=norm)
    x = fft_torch.fftshift(x, dim=dim)
    return x

def ifft(x, dim=None, oshape=None, norm='ortho'):
    """Matches Sigpy's fft adjoint, but in torch"""

    if oshape is not None:
        x_cp = torch_to_np(x)
        dev = sp.get_device(x_cp)
        with dev:
            x = np_to_torch(sp.resize(x_cp, oshape))
    x = fft_torch.ifftshift(x, dim=dim)
    x = fft_torch.ifftn(x, dim=dim, norm=norm)
    x = fft_torch.fftshift(x, dim=dim)
    return x

def calc_toep_kernel_helper(nufft_adj_os: callable,
                            trj: torch.Tensor,
                            weights: Optional[torch.Tensor] = None):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        nufft_adj_os : callable
            Performs adjoint NUFFT to oversampled image shape (*im_size_os)
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float32>
            weighting function with shape (N, *trj_batch)

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex64>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        torch_dev = trj.device
        if weights is None:
            weights = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert weights.device == torch_dev
        trj_batch = trj.shape[1:-1]
        if trj.shape[0] == 1 and weights.shape[0] != 1:
            trj = trj.expand((weights.shape[0], *trj.shape[1:]))
        N = trj.shape[0]
        d = trj.shape[-1]
        
        # Get toeplitz kernel via adjoint nufft on 1s ksp
        ksp = torch.ones((N, 1, *trj_batch), device=torch_dev, dtype=torch.complex64)
        ksp_weighted = ksp * weights[:, None, ...]
        img = nufft_adj_os(ksp_weighted, trj)[:, 0, ...] # (N, *im_size_os)

        # FFT
        toeplitz_kernels = fft(img, dim=tuple(range(-d, 0)))

        return toeplitz_kernels

class NUFFT(nn.Module):

    def __init__(self,
                 im_size: tuple,
                 device_idx: Optional[int] = None):
        """
        Parameters:
        -----------
        im_size : tuple
            image dimensions
        device_idx : int
            index of GPU device, -1 for CPU
        """
        super(NUFFT, self).__init__()
        self.im_size = im_size
        if device_idx is None or device_idx == -1:
            self.torch_dev = torch.device('cpu')
        else:
            self.torch_dev = torch.device(device_idx)
    
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        """
        Different NUFFT options may have different desired 
        trajectory scalings/dimensions, handeled here.

        Parameters:
        -----------
        trj : torch.Tensor <float32>
            input trajectory shape (..., d) where d = 2 or 3 for 2D/3D
        
        Returns:
        --------
        trj_rs : torch.Tensor <float32>
            the rescaled trajectory with shape (..., d)
        """

        return trj

    def forward(self,
                img: torch.Tensor,
                trj: torch.Tensor) -> torch.Tensor:
        """
        Non-unfiform fourier transform

        Parameters:
        -----------
        img : torch.Tensor <complex64>
            input image with shape (N, *img_batch, *im_size)
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))

        Returns:
        --------
        ksp : torch.Tensor <complex64>
            output k-space with shape (N, *img_batch, *trj_batch)

        Note:
        -----
        N is the batch dim, must pass in 1 if no batching!
        """
        raise NotImplementedError
    
    def adjoint(self,
                ksp: torch.Tensor,
                trj: torch.Tensor) -> torch.Tensor:
        """
        Adjoint non-uniform fourier transform

        Parameters:
        -----------
        ksp : torch.Tensor <complex64>
            input k-space with shape (N, *ksp_batch, *trj_batch)
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        img : torch.Tensor <complex64>
            output image with shape (N, *ksp_batch, *im_size)
        
        Note:
        -----
        N is the batch dim, must pass in 1 if no batching!
        """
        raise NotImplementedError

    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float32>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex64>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """
        raise NotImplementedError

    def normal_toeplitz(self,
                        img: torch.Tensor,
                        toeplitz_kernels: torch.Tensor) -> torch.Tensor:
        """
        NUFFT_adjoint NUFFT operation using pre-calculated Toeplitz kernels

        Parameters:
        -----------
        img : torch.Tensor <complex64>
            input image with shape (N, *img_batch, *im_size)
        toeplitz_kernels : torch.Tensor <complex64>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size

        Returns:
        --------
        img_hat : torch.Tensor <complex64>
            output image with shape (N, *img_batch, *im_size)
        """
        
        # Consts
        N = img.shape[0]
        im_size = self.im_size
        im_size_os = toeplitz_kernels.shape[1:]
        d = len(im_size)
        img_batch = img.shape[1:-d]
        img_flt = img.reshape((N, -1, *im_size))
        n_batch_size = 1
        img_batch_size = 1
        
        # Make padder 
        padder = PadLast(im_size_os, im_size)

        # Output image
        img_hat_flt = torch.zeros_like(img_flt)

        # batching loops
        for n1, n2 in batch_iterator(N, n_batch_size):
            for i1, i2 in batch_iterator(img.shape[1], img_batch_size):
                frwrd = img[n1:n2, i1:i2, ...]
                frwrd = padder.forward(frwrd)
                frwrd = fft(frwrd, dim=tuple(range(-d, 0)))
                frwrd = frwrd * toeplitz_kernels[n1:n2, None, ...]
                frwrd = ifft(frwrd, dim=tuple(range(-d, 0)))
                frwrd = padder.adjoint(frwrd)
                img_hat_flt[n1:n2, i1:i2, ...] = frwrd

        # Reshape and return
        img_hat = img_hat_flt.reshape((N, *img_batch, *im_size))

        return img_hat

class sigpy_nufft(NUFFT):
    
    def __init__(self,
                 im_size: tuple,
                 device_idx: Optional[int] = -1,
                 oversamp: Optional[float] = 1.25,
                 width: Optional[int] = 6):
        super().__init__(im_size, device_idx)
        self.oversamp = oversamp
        self.width = width
    
    def forward(self, 
                img: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        """
        Figured I should explain myself before commiting this code crime.
        I want to batch the NUFFT, and this can be done by modifying the 
        final KB interpolation step. So I am just going to copy the sigpy
        NUFFT implimentation, which feels so very wrong, and then augment 
        the KB inerpolate part :')
        """

        # Convert to cupy first
        img_cp, trj_cp = torch_to_np(img, trj)
        
        # Consts
        width = self.width
        oversamp = self.oversamp
        beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
        ndim = trj_cp.shape[-1]
        os_shape = _get_oversamp_shape(img_cp.shape, ndim, oversamp)
        N = trj.shape[0]

        dev = sp.get_device(trj_cp)
        with dev:
        
            ksp = img_cp.copy()

            # Apodize
            _apodize(ksp, ndim, oversamp, width, beta)

            # Zero-pad
            ksp /= sp.util.prod(img_cp.shape[-ndim:])**0.5
            ksp = sp.util.resize(ksp, os_shape)

            # FFT
            # ksp = sp.fft(ksp, axes=range(-ndim, 0), norm=None)
            ksp = torch_to_np(fft(np_to_torch(ksp), dim=tuple(range(-ndim, 0)), norm=None))

            # Interpolate
            trj_cp = _scale_coord(trj_cp, img_cp.shape, oversamp)
            ksp_ret = dev.xp.zeros((N, *img_cp.shape[1:-ndim], *trj.shape[1:-1]), dtype=dev.xp.complex64)
            for i in range(N):
                ksp_ret[i] = sp.interp.interpolate(
                        ksp[i], trj_cp[i], kernel='kaiser_bessel', width=width, param=beta)
            ksp_ret /= width ** ndim
        return np_to_torch(ksp_ret)

    def adjoint(self, 
                ksp: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        
        # Convert to cupy first
        ksp_cp, trj_cp = torch_to_np(ksp, trj)
        
        # Consts
        width = self.width
        oversamp = self.oversamp
        beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
        ndim = trj_cp.shape[-1]
        N = trj.shape[0]
        im_size = self.im_size
        oshape = (N, *ksp.shape[1:-(trj.ndim-2)], *im_size)
        os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

        # Gridding
        dev = sp.get_device(trj_cp)
        with dev:
            trj_cp = _scale_coord(trj_cp, oshape, oversamp)
            output = dev.xp.zeros(os_shape, dtype=dev.xp.complex64)
            for i in range(N):
                output[i] = sp.interp.gridding(ksp_cp[i], trj_cp[i], os_shape[1:], 
                                                 kernel='kaiser_bessel', width=width, param=beta)
            output /= width**ndim

            # IFFT
            # output = sp.ifft(output, axes=range(-ndim, 0), norm=None)
            output = torch_to_np(ifft(np_to_torch(output), dim=tuple(range(-ndim, 0)), norm=None))

            # Crop
            output = sp.util.resize(output, oshape)
            output *= sp.util.prod(os_shape[-ndim:]) / sp.util.prod(oshape[-ndim:])**0.5

            # Apodize
            _apodize(output, ndim, oversamp, width, beta)

        return np_to_torch(output)

    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float32>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex64>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = sigpy_nufft(im_size=im_size_os, device_idx=self.torch_dev.index, 
                               oversamp=self.oversamp, width=self.width)

        return calc_toep_kernel_helper(nufft_os.adjoint, trj * os_factor, weights) * (os_factor ** len(self.im_size))

class torchkb_nufft(NUFFT):

    def __init__(self,
                 im_size: tuple,
                 device_idx: Optional[int] = -1,
                 oversamp: Optional[float] = 2.0,
                 numpoints: Optional[int] = 6):
        super().__init__(im_size, device_idx)
        
        im_size_os = tuple([round(i * oversamp) for i in im_size])
        self.kb_ob = KbNufft(im_size, device=self.torch_dev, grid_size=im_size_os, numpoints=numpoints).to(self.torch_dev)
        self.kb_adj_ob = KbNufftAdjoint(im_size, device=self.torch_dev, grid_size=im_size_os, numpoints=numpoints).to(self.torch_dev)
        self.oversamp = oversamp
        self.numpoints = numpoints

    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Rescale to -pi, pi
        im_size_arr = torch.tensor(self.im_size).to(trj.device)
        tup = (None,) * (trj.ndim - 1) + (slice(None),)
        trj_rs = torch.pi * trj / (im_size_arr[tup] / 2)

        return trj_rs

    def forward(self, 
                img: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        
        # To torch
        img_torch, trj_torch = np_to_torch(img, trj)
        im_size = self.im_size
        N = trj.shape[0]
        d = trj.shape[-1]

        # Reshape - NUFFT - Reshape
        img_torchkb = img_torch.reshape((N, -1, *im_size))
        omega = trj_torch.reshape((N, -1, d)).swapaxes(-2, -1)
        ksp = self.kb_ob(image=img_torchkb, omega=omega, norm='ortho')
        ksp = ksp.reshape((N, *img.shape[1:-d], *trj.shape[1:-1]))
        return ksp * self.oversamp

    def adjoint(self,
                ksp: torch.Tensor,
                trj: torch.Tensor) -> torch.Tensor:

        # To torch
        ksp_torch, trj_torch = np_to_torch(ksp, trj)
        im_size = self.im_size
        N = trj.shape[0]
        d = trj.shape[-1]

        # Reshape - NUFFT - Reshape
        ksp_torch_kb = ksp_torch.reshape((N, -1, *trj.shape[1:-1]))
        ksp_torch_kb = ksp_torch_kb.reshape((N, ksp_torch_kb.shape[1], -1))
        omega = trj_torch.reshape((N, -1, d)).swapaxes(-2, -1)
        img = self.kb_adj_ob(data=ksp_torch_kb, omega=omega, norm='ortho')
        img = img.reshape((N, *ksp.shape[1:-(trj.ndim - 2)], *im_size))
        return img * self.oversamp
    
    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float32>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex64>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = torchkb_nufft(im_size_os, device_idx=self.torch_dev.index, oversamp=self.oversamp, numpoints=self.numpoints)

        return calc_toep_kernel_helper(nufft_os.adjoint, trj, weights) * (os_factor ** len(self.im_size))
    
class gridded_nufft(NUFFT):

    def __init__(self,
                 im_size: tuple,
                 device_idx: Optional[int] = -1,
                 grid_oversamp: Optional[float] = 1.0):
        super().__init__(im_size, device_idx)
        grog_os_size = tuple([round(i * grid_oversamp) for i in self.im_size])
        self.grid_oversamp = grid_oversamp
        self.grog_padder = PadLast(grog_os_size, im_size)
    
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Clamp each dimension
        trj_rs = trj * self.grid_oversamp
        grid_os_size = self.grog_padder.pad_im_size
        for i in range(trj_rs.shape[-1]):
            n_over_2 = grid_os_size[i]/2
            trj_rs[..., i] = torch.clamp(trj_rs[..., i] + n_over_2, 0, grid_os_size[i]-1)
        trj_rs = torch.round(trj_rs).type(torch.int32)

        return trj_rs

    def forward(self, 
                img: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        
        # To torch
        img_torch, trj_torch = np_to_torch(img, trj)

        # Consts
        d = trj.shape[-1]
        N = trj.shape[0]

        # Oversampled FFT
        img_os = self.grog_padder.forward(img_torch)
        ksp_os = fft(img_os, dim=tuple(range(-d, 0)))
        
        # Return k-space
        ksp = torch.zeros((*img.shape[:-d], *trj.shape[1:-1]), 
                          dtype=torch.complex64, device=img_torch.device)
        for i in range(N):
            ksp[i] = multi_index(ksp_os[i], d, trj_torch[i].type(torch.int32))
        
        return ksp * self.grid_oversamp
                    
    def adjoint(self, 
                ksp: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        """
        Adjoint non-uniform fourier transform

        Parameters:
        -----------
        ksp : torch.Tensor <complex64>
            input k-space with shape (N, *ksp_batch, *trj_batch)
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        img : torch.Tensor <complex64>
            output image with shape (N, *ksp_batch, *im_size)
        
        Note:
        -----
        N is the batch dim, must pass in 1 if no batching!
        """

        # To torch
        ksp_torch, trj_torch = np_to_torch(ksp, trj)
        
        # Consts
        d = trj.shape[-1]
        N = trj.shape[0]
        grid_os_size = self.grog_padder.pad_im_size

        # Adjoint NUFFT
        ksp_os = torch.zeros((*ksp.shape[:-(trj.ndim - 2)], *grid_os_size), 
                             dtype=torch.complex64, device=ksp_torch.device)
        for i in range(N):
            ksp_os[i] = multi_grid(ksp_torch[i], trj_torch[i].type(torch.int32), grid_os_size)
        img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
        img = self.grog_padder.adjoint(img_os)

        return img * self.grid_oversamp
    
    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = None):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float32>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz (unused here)

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex64>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        os_factor = self.grid_oversamp
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = gridded_nufft(im_size_os, device_idx=self.torch_dev.index, grid_oversamp=self.grid_oversamp)

        return calc_toep_kernel_helper(nufft_os.adjoint, (trj * os_factor).type(torch.int32), weights) * (os_factor ** len(self.im_size))
