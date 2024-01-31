import torch
import torch.nn as nn
import numpy as np
import sigpy as sp

from typing import Optional
from torchkbnufft import KbNufft, KbNufftAdjoint
from mr_recon.utils.pad import PadLast
from mr_recon.utils.indexing import (
    multi_grid,
    multi_index
)
from sigpy.fourier import (
    _get_oversamp_shape, 
    _apodize, 
    _scale_coord)
from mr_recon.utils.func import (
    torch_to_np, 
    np_to_torch,
    sp_fft,
    sp_ifft)

class NUFFT(nn.Module):

    def __init__(self,
                 im_size: tuple,
                 device_idx: Optional[int] = -1):
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
        if device_idx == -1:
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
            ksp = sp.fft(ksp, axes=range(-ndim, 0), norm=None)

            # Interpolate
            trj_cp = _scale_coord(trj_cp, img_cp.shape, oversamp)
            ksp_ret = dev.xp.zeros((N, *img_cp.shape[1:-ndim], *trj.shape[1:-1]), dtype=dev.xp.complex64)
            for i in range(N):
                ksp_ret[i] = sp.interp.interpolate(
                        ksp[i], trj_cp[i], kernel='kaiser_bessel', width=width, param=beta)
            ksp_ret /= width ** ndim
        return np_to_torch(ksp_ret)[0]

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
            output = sp.ifft(output, axes=range(-ndim, 0), norm=None)

            # Crop
            output = sp.util.resize(output, oshape)
            output *= sp.util.prod(os_shape[-ndim:]) / sp.util.prod(oshape[-ndim:])**0.5

            # Apodize
            _apodize(output, ndim, oversamp, width, beta)

        return np_to_torch(output)[0]

class torchkb_nufft(NUFFT):

    def __init__(self,
                 im_size: tuple,
                 device_idx: Optional[int] = -1):
        super().__init__(im_size, device_idx)

        self.kb_ob = KbNufft(im_size, device=self.torch_dev).to(self.torch_dev)
        self.kb_adj_ob = KbNufftAdjoint(im_size, device=self.torch_dev).to(self.torch_dev)

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
        return ksp

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
        return img
    
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
        assert img.shape[0] == trj.shape[0], "Batch size mismatch!"

        # To torch
        img_torch, trj_torch = np_to_torch(img, trj)

        # Consts
        d = trj.shape[-1]
        N = trj.shape[0]

        # Oversampled FFT
        img_os = self.grog_padder.forward(img_torch)
        ksp_os = sp_fft(img_os, dim=tuple(range(-d, 0)))
        
        # Return k-space
        ksp = torch.zeros((*img.shape[:-d], *trj.shape[1:-1]), 
                          dtype=torch.complex64, device=img_torch.device)
        for i in range(N):
            ksp[i] = multi_index(ksp_os[i], d, trj_torch[i])
        
        return ksp
                    
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
            ksp_os[i] = multi_grid(ksp_torch[i], trj_torch[i], grid_os_size)
        img_os = sp_ifft(ksp_os, dim=tuple(range(-d, 0)))
        img = self.grog_padder.adjoint(img_os)

        return img