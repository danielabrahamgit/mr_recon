import gc
import torch
import torch.nn as nn
import numpy as np
import sigpy as sp
import torch.fft as fft_torch

from typing import Optional
from torchkbnufft import KbNufft, KbNufftAdjoint
from einops import einsum, rearrange
from scipy.special import jv
from cupyx.scipy.ndimage import map_coordinates
from math import ceil, floor
from mr_recon import dtypes
from mr_recon.pad import PadLast
from mr_recon.algs import svd_power_method_tall, eigen_decomp_operator
from mr_recon.triton_interp.interp import interpolate, interpolate_adjoint
from mr_recon.triton_interp.ungrid import ungrid, ungrid_torch
from mr_recon.indexing import (
    multi_grid,
    multi_index
)
from sigpy.fourier import (
    _get_oversamp_shape, 
    _apodize, 
    _scale_coord)
from mr_recon.utils import (
    gen_grd,
    torch_to_np, 
    np_to_torch,
    batch_iterator,
    resize)

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
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float>
            weighting function with shape (N, *trj_batch)

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        torch_dev = trj.device
        if weights is None:
            weights = torch.ones(trj.shape[:-1], dtype=dtypes.real_dtype, device=torch_dev)
        else:
            assert weights.device == torch_dev
        trj_batch = trj.shape[1:-1]
        if trj.shape[0] == 1 and weights.shape[0] != 1:
            trj = trj.expand((weights.shape[0], *trj.shape[1:]))
        N = trj.shape[0]
        d = trj.shape[-1]
        
        # Get toeplitz kernel via adjoint nufft on 1s ksp
        ksp = torch.ones((N, 1, *trj_batch), device=torch_dev, dtype=dtypes.complex_dtype)
        ksp_weighted = ksp * weights[:, None, ...]
        img = nufft_adj_os(ksp_weighted, trj)[:, 0, ...] # (N, *im_size_os)

        # FFT
        toeplitz_kernels = fft(img, dim=tuple(range(-d, 0)))

        return toeplitz_kernels

def _torch_apodize(input, ndim, oversamp, width, beta):
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = ceil(oversamp * i)
        idx = torch.arange(i, device=output.device)

        # Calculate apodization
        apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
        apod /= torch.sinh(apod)
        output *= apod.reshape([i] + [1] * (-a - 1))

    return output

class NUFFT(nn.Module):
    """
    Forward NUFFT is defined as:
    e^{-j2\pi k \cdot r}

    where k_i \in [-N_/2, N_i/2] and im_size[i] = N_i
    and r_i \in [-1/2, 1/2].
    """

    def __init__(self,
                 im_size: tuple):
        """
        Parameters:
        -----------
        im_size : tuple
            image dimensions
        """
        super(NUFFT, self).__init__()
        self.im_size = im_size

    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        """
        Different NUFFT options may have different desired 
        trajectory scalings/dimensions, handeled here.

        Parameters:
        -----------
        trj : torch.Tensor <float>
            input trajectory shape (..., d) where d = 2 or 3 for 2D/3D
        
        Returns:
        --------
        trj_rs : torch.Tensor <float>
            the rescaled trajectory with shape (..., d)
        """

        return trj

    def forward_FT_only(self,
                        img: torch.Tensor) -> torch.Tensor:
        """
        Only does the FT part of the nufft. This includes
        - apodization
        - zero padding
        - fft
        
        Parameters:
        -----------
        img : torch.Tensor <complex>
            input image with shape (N, *img_batch, *im_size)
        
        Returns:
        --------
        ksp_os : torch.Tensor <complex>
            k-space grid with shape (N, *img_batch, *im_size_os)
        """
        raise NotImplementedError
    
    def adjoint_iFT_only(self,
                         ksp_os: torch.Tensor) -> torch.Tensor:
        """
        Only does the iFT part of the nufft. This includes
        - iFFT
        - Cropping
        - Apodization
        
        Parameters:
        -----------
        ksp_os : torch.Tensor <complex>
            k-space grid with shape (N, *img_batch, *im_size_os)
        
        Returns:
        --------
        img : torch.Tensor <complex>
            output image with shape (N, *img_batch, *im_size)
        """
        raise NotImplementedError
    
    def adjoint_grid_only(self,
                          ksp: torch.Tensor,
                          trj: torch.Tensor) -> torch.Tensor:
        """
        Only does the gridding part of the nufft. This includes
        - gridding
        - scaling
        
        Parameters:
        -----------
        ksp : torch.Tensor <complex>
            input k-space with shape (N, *ksp_batch, *trj_batch)
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        ksp_os : torch.Tensor <complex>
            output k-space with shape (N, *ksp_batch, *im_size_os)
        """
        raise NotImplementedError
    
    def forward_interp_only(self,
                            ksp_os: torch.Tensor,
                            trj: torch.Tensor) -> torch.Tensor:
        """
        Only does the interpolation part of the nufft. This includes
        - interpolation
        - scaling
        
        Parameters:
        -----------
        ksp_os : torch.Tensor <complex>
            k-space grid with shape (N, *img_batch, *im_size_os)
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        ksp : torch.Tensor <complex>
            k-space with shape (N, *img_batch, *trj_batch)
        """        
        raise NotImplementedError

    def forward(self,
                img: torch.Tensor,
                trj: torch.Tensor) -> torch.Tensor:
        """
        Non-unfiform fourier transform

        Parameters:
        -----------
        img : torch.Tensor <complex>
            input image with shape (N, *img_batch, *im_size)
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))

        Returns:
        --------
        ksp : torch.Tensor <complex>
            output k-space with shape (N, *img_batch, *trj_batch)

        Note:
        -----
        N is the batch dim, must pass in 1 if no batching!
        """
        return self.forward_interp_only(self.forward_FT_only(img), trj)
    
    def adjoint(self,
                ksp: torch.Tensor,
                trj: torch.Tensor) -> torch.Tensor:
        """
        Adjoint non-uniform fourier transform

        Parameters:
        -----------
        ksp : torch.Tensor <complex>
            input k-space with shape (N, *ksp_batch, *trj_batch)
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        img : torch.Tensor <complex>
            output image with shape (N, *ksp_batch, *im_size)
        
        Note:
        -----
        N is the batch dim, must pass in 1 if no batching!
        """
        return self.adjoint_iFT_only(self.adjoint_grid_only(ksp, trj))

    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex>
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
        img : torch.Tensor <complex>
            input image with shape (N, *img_batch, *im_size)
        toeplitz_kernels : torch.Tensor <complex>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size

        Returns:
        --------
        img_hat : torch.Tensor <complex>
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

class triton_nufft(NUFFT):
    """
    Uses Mark Nishimura's Triton implimentation of gridding and interpolation 
    to perform the NUFFT:
    https://github.com/nishi951/torch-named-linops/blob/main/src/torchlinops/functional/_interp/interp.py
    """
    
    def __init__(self,
                 im_size: tuple,
                 oversamp: Optional[float] = 1.25,
                 width: Optional[int] = 6,
                 beta: Optional[float] = None,
                 apodize: Optional[bool] = True):
        super().__init__(im_size)
        self.oversamp = oversamp
        self.width = width
        self.apodize = apodize
        if beta is None:
            self.beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
        else:
            self.beta = beta
    
    def _apodization_func(self, 
                          x: torch.Tensor) -> torch.Tensor:
        """
        1D apodization for the NUFFT
        
        Parameters:
        -----------
        x : torch.Tensor <float>
            Arb shape signal between [-1/2, 1/2]
            
        Returns:
        --------
        apod : torch.Tensor <float>
            apodization evaluated at input x
        """

        apod = (
            self.beta**2 - (np.pi * self.width * x) ** 2
        ) ** 0.5
        apod /= torch.sinh(apod)
        return apod
    
    def apodize_img(self,
                    img: torch.Tensor) -> torch.Tensor:
        """
        Applies apodization to the input image
        
        Parameters:
        -----------
        img : torch.Tensor <complex>
            input image with shape (..., *im_size)

        Returns:
        --------
        img_apod : torch.Tensor <complex>
            apodized image with shape (..., *im_size)
        """
        ndim = len(self.im_size)
        for i in range(-ndim, 0):
            crds = torch.arange(-(img.shape[i] // 2), img.shape[i] // 2, device=img.device)
            tup = (slice(None),) + (None,) * (-i-1)
            img *= self._apodization_func(crds / ceil(self.oversamp * img.shape[i]))[tup]
        return img
    
    def _scale_trj(self, trj, shape):
        ndim = trj.shape[-1]
        output = trj.clone()
        oversamp = self.oversamp
        for i in range(-ndim, 0):
            scale = ceil(oversamp * shape[i]) / shape[i]
            shift = ceil(oversamp * shape[i]) // 2
            output[..., i] *= scale
            output[..., i] += shift

        return output
    
    def forward_FT_only(self, 
                        img: torch.Tensor) -> torch.Tensor:
        """
        Only does the FT part of the nufft. This includes
        - apodization 
        - zero padding 
        - fft 

        Parameters:
        -----------
        img : torch.Tensor <complex>
            input image with shape (N, *img_batch, *im_size)
        
        Returns:
        --------
        ksp_os : torch.Tensor <complex>
            k-space grid with shape (N, *img_batch, *im_size_os)
        """
        
        # Consts
        oversamp = self.oversamp
        ndim = len(self.im_size)
        os_shape = list(img.shape)[:-ndim] + [ceil(oversamp * i) for i in img.shape[-ndim:]]

        # Apodize
        img_apod = img.clone()
        if self.apodize:
            img_apod = self.apodize_img(img_apod)

        # Zero-pad
        img_apod /= np.prod(img_apod.shape[-ndim:])**0.5
        img_zp = resize(img_apod, os_shape)

        # FFT
        ksp_os = fft(img_zp, dim=tuple(range(-ndim, 0)), norm=None)

        return ksp_os

    def forward_interp_only(self,
                            ksp_os: torch.Tensor,
                            trj: torch.Tensor) -> torch.Tensor:
        """
        Only does the interpolation part of the nufft. Input is output of forward_FT_only.

        Parameters:
        -----------
        ksp_os : torch.Tensor <complex>
            k-space grid with shape (N, *img_batch, *im_size_os)

        Returns:
        --------
        ksp : torch.Tensor <complex>
            k-space with shape (N, *img_batch, *trj_size)
        """

        # Consts
        width = self.width
        beta = self.beta
        ndim = len(self.im_size)
        img_shape = (*ksp_os.shape[:-ndim], *self.im_size)
        N = trj.shape[0]    

        # Interpolate
        trj = self._scale_trj(trj, img_shape)
        ksp_ret = torch.zeros((N, *img_shape[1:-ndim], *trj.shape[1:-1]), dtype=complex_dtype, device=ksp_os.device)
        for i in range(N):
            ksp_ret[i] = interpolate(ksp_os[i], trj[i], width=(width,)*ndim, 
                                     kernel='kaiser_bessel', 
                                     kernel_params={'beta':beta}, 
                                     pad_mode='circular')
        ksp_ret /= width ** ndim
        return np_to_torch(ksp_ret)
        
    def adjoint_iFT_only(self,
                         ksp_os: torch.Tensor) -> torch.Tensor:
        # Consts
        oversamp = self.oversamp
        ndim = len(self.im_size)
        im_size = self.im_size
        oshape = (*ksp_os.shape[:-ndim], *im_size)
        os_shape = list(oshape)[:-ndim] + [ceil(oversamp * i) for i in oshape[-ndim:]]
        
        # iFFT
        img_os = ifft(ksp_os, dim=tuple(range(-ndim, 0)), norm=None)

        # Crop
        output = resize(img_os, oshape)
        output *= np.prod(os_shape[-ndim:]) / np.prod(oshape[-ndim:])**0.5

        # Apodize
        if self.apodize:
            output = self.apodize_img(output)

        return output
    
    def adjoint_grid_only(self,
                          ksp: torch.Tensor, 
                          trj: torch.Tensor):
        
        # Consts
        width = self.width
        oversamp = self.oversamp
        beta = self.beta
        ndim = trj.shape[-1]
        N = trj.shape[0]
        im_size = self.im_size
        oshape = (N, *ksp.shape[1:-(trj.ndim-2)], *im_size)
        os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

        # Gridding
        trj = self._scale_trj(trj, oshape)
        output = torch.zeros(os_shape, dtype=complex_dtype, device=ksp.device)
        for i in range(N):
            output[i] = interpolate_adjoint(ksp[i], trj[i], os_shape[-ndim:], 
                                            kernel='kaiser_bessel', width=(width,)*ndim, kernel_params={'beta':beta},
                                            pad_mode='circular')
        output /= width**ndim
    
        return output

    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = triton_nufft(im_size=im_size_os, 
                               oversamp=self.oversamp, width=self.width)

        return calc_toep_kernel_helper(nufft_os.adjoint, trj * os_factor, weights) * (os_factor ** len(self.im_size))

class sigpy_nufft(NUFFT):
    
    def __init__(self,
                 im_size: tuple,
                 oversamp: Optional[float] = 1.25,
                 width: Optional[int] = 6,
                 beta: Optional[float] = None,
                 apodize: Optional[bool] = True):
        super().__init__(im_size)
        self.oversamp = oversamp
        self.width = width
        self.apodize = apodize
        if beta is None:
            self.beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
        else:
            self.beta = beta
    
    def _apodization_func(self, 
                          x: torch.Tensor) -> torch.Tensor:
        """
        1D apodization for the NUFFT
        
        Parameters:
        -----------
        x : torch.Tensor <float>
            Arb shape signal between [-1/2, 1/2]
            
        Returns:
        --------
        apod : torch.Tensor <float>
            apodization evaluated at input x
        """

        apod = (
            self.beta**2 - (np.pi * self.width * x) ** 2
        ) ** 0.5
        apod /= torch.sinh(apod)
        return apod
    
    def _scale_trj(self, trj, shape):
        ndim = trj.shape[-1]
        output = trj.clone()
        oversamp = self.oversamp
        for i in range(-ndim, 0):
            scale = ceil(oversamp * shape[i]) / shape[i]
            shift = ceil(oversamp * shape[i]) // 2
            output[..., i] *= scale
            output[..., i] += shift

        return output
    
    def apodize_img(self,
                    img: torch.Tensor) -> torch.Tensor:
        """
        Applies apodization to the input image
        
        Parameters:
        -----------
        img : torch.Tensor <complex>
            input image with shape (..., *im_size)

        Returns:
        --------
        img_apod : torch.Tensor <complex>
            apodized image with shape (..., *im_size)
        """
        ndim = len(self.im_size)
        for i in range(-ndim, 0):
            crds = torch.arange(img.shape[i], device=img.device) - img.shape[i] // 2
            tup = (slice(None),) + (None,) * (-i-1)
            img *= self._apodization_func(crds / ceil(self.oversamp * img.shape[i]))[tup]
        return img
    
    def forward_FT_only(self, 
                        img: torch.Tensor) -> torch.Tensor:
        """
        Only does the FT part of the nufft. This includes
        - apodization 
        - zero padding 
        - fft 

        Parameters:
        -----------
        img : torch.Tensor <complex>
            input image with shape (N, *img_batch, *im_size)
        
        Returns:
        --------
        ksp_os : torch.Tensor <complex>
            k-space grid with shape (N, *img_batch, *im_size_os)
        """
        # Consts
        oversamp = self.oversamp
        ndim = len(self.im_size)
        os_shape = list(img.shape)[:-ndim] + [ceil(oversamp * i) for i in img.shape[-ndim:]]

        # Apodize
        img_apod = img.clone()
        if self.apodize:
            img_apod = self.apodize_img(img_apod)

        # Zero-pad
        img_apod /= np.prod(img_apod.shape[-ndim:])**0.5
        img_zp = resize(img_apod, os_shape)

        # FFT
        ksp_os = fft(img_zp, dim=tuple(range(-ndim, 0)), norm=None)

        return ksp_os

    def forward_interp_only(self,
                            ksp_os: torch.Tensor,
                            trj: torch.Tensor) -> torch.Tensor:
        """
        Only does the interpolation part of the nufft. Input is output of forward_FT_only.

        Parameters:
        -----------
        ksp_os : torch.Tensor <complex>
            k-space grid with shape (N, *img_batch, *im_size_os)

        Returns:
        --------
        ksp : torch.Tensor <complex>
            k-space with shape (N, *img_batch, *trj_size)
        """

        # Consts
        width = self.width
        beta = self.beta
        ndim = len(self.im_size)
        img_shape = (*ksp_os.shape[:-ndim], *self.im_size)
        N = trj.shape[0]
        
        # Scale trajectory and move to cupy
        trj_cp = torch_to_np(self._scale_trj(trj, img_shape))
        ksp_os_cp = torch_to_np(ksp_os)

        # Interpolate
        dev = sp.get_device(ksp_os_cp)
        with dev:
            trj_cp = _scale_coord(trj_cp, img_shape, oversamp)
            ksp_ret = dev.xp.zeros((N, *img_shape[1:-ndim], *trj.shape[1:-1]), dtype=dtypes.np_complex_dtype)
            for i in range(N):
                ksp_ret[i] = sp.interp.interpolate(
                        ksp_os_cp[i], trj_cp[i], kernel='kaiser_bessel', width=width, param=beta)
            ksp_ret /= width ** ndim
        return np_to_torch(ksp_ret)

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
        return self.forward_interp_only(self.forward_FT_only(img), trj)
        # Convert to cupy first
        img_cp, trj_cp = torch_to_np(img, trj)
        
        # Consts
        width = self.width
        oversamp = self.oversamp
        beta = self.beta
        ndim = trj_cp.shape[-1]
        os_shape = _get_oversamp_shape(img_cp.shape, ndim, oversamp)
        N = trj.shape[0]

        dev = sp.get_device(trj_cp)
        with dev:
        
            ksp = img_cp.copy()

            # Apodize
            if self.apodize:
                _apodize(ksp, ndim, oversamp, width, beta)

            # Zero-pad
            ksp /= sp.util.prod(img_cp.shape[-ndim:])**0.5
            ksp = sp.util.resize(ksp, os_shape)

            # FFT
            ksp = torch_to_np(fft(np_to_torch(ksp), dim=tuple(range(-ndim, 0)), norm=None))

            # Interpolate
            trj_cp = _scale_coord(trj_cp, img_cp.shape, oversamp)
            ksp_ret = dev.xp.zeros((N, *img_cp.shape[1:-ndim], *trj.shape[1:-1]), dtype=dtypes.np_complex_dtype)
            for i in range(N):
                ksp_ret[i] = sp.interp.interpolate(
                        ksp[i], trj_cp[i], kernel='kaiser_bessel', width=width, param=beta)
            ksp_ret /= width ** ndim
        return np_to_torch(ksp_ret)

    def adjoint_grid_only(self,
                          ksp: torch.Tensor, 
                          trj: torch.Tensor):
        # Convert to cupy first
        ksp_cp, trj_cp = torch_to_np(ksp, trj)
        
        # Consts
        width = self.width
        oversamp = self.oversamp
        beta = self.beta
        ndim = trj_cp.shape[-1]
        N = trj.shape[0]
        im_size = self.im_size
        oshape = (N, *ksp.shape[1:-(trj.ndim-2)], *im_size)
        os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

        # Gridding
        dev = sp.get_device(trj_cp)
        with dev:
            trj_cp = _scale_coord(trj_cp, oshape, oversamp)
            output = dev.xp.zeros(os_shape, dtype=dtypes.np_complex_dtype)
            for i in range(N):
                output[i] = sp.interp.gridding(ksp_cp[i], trj_cp[i], os_shape[1:], 
                                                 kernel='kaiser_bessel', width=width, param=beta)
            output /= width**ndim
    
        return np_to_torch(output)

    def adjoint_iFT_only(self,
                         ksp_os: torch.Tensor) -> torch.Tensor:
        # Consts
        oversamp = self.oversamp
        ndim = len(self.im_size)
        im_size = self.im_size
        oshape = (*ksp_os.shape[:-ndim], *im_size)

        os_shape = list(oshape)[:-ndim] + [ceil(oversamp * i) for i in oshape[-ndim:]]
        
        # iFFT
        img_os = ifft(ksp_os, dim=tuple(range(-ndim, 0)), norm=None)
        
        # Crop
        output = resize(img_os, oshape)
        output *= np.prod(os_shape[-ndim:]) / np.prod(oshape[-ndim:])**0.5
        
        # Apodize
        if self.apodize:
            output = self.apodize_img(output)
        return output

    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = sigpy_nufft(im_size=im_size_os, 
                               oversamp=self.oversamp, width=self.width)

        return calc_toep_kernel_helper(nufft_os.adjoint, trj * os_factor, weights) * (os_factor ** len(self.im_size))

class torchkb_nufft(NUFFT):

    def __init__(self,
                 im_size: tuple,
                 torch_dev: Optional[torch.device] = torch.device('cpu'),
                 oversamp: Optional[float] = 2.0,
                 numpoints: Optional[int] = 6):
        super().__init__(im_size)
        
        im_size_os = tuple([round(i * oversamp) for i in im_size])
        self.kb_ob = KbNufft(im_size, device=torch_dev, grid_size=im_size_os, numpoints=numpoints).to(torch_dev)
        self.kb_adj_ob = KbNufftAdjoint(im_size, device=torch_dev, grid_size=im_size_os, numpoints=numpoints).to(torch_dev)
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
        return ksp * np.sqrt(self.oversamp) # NOTE: scaling updated as sqrt to match sigpy scaling

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
        return img * np.sqrt(self.oversamp)
    
    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = torchkb_nufft(im_size_os, torch_dev=trj.device, oversamp=self.oversamp, numpoints=self.numpoints)

        return calc_toep_kernel_helper(nufft_os.adjoint, trj, weights) * (os_factor ** len(self.im_size))
    
class gridded_nufft(NUFFT):

    def __init__(self,
                 im_size: tuple,
                 grid_oversamp: Optional[float] = 1.0):
        super().__init__(im_size)
        self.im_size_os = tuple([round(i * grid_oversamp) for i in self.im_size])
        self.grid_oversamp = grid_oversamp
    
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Clamp each dimension
        trj_rs = trj * self.grid_oversamp
        for i in range(trj_rs.shape[-1]):
            n_over_2 = self.im_size_os[i]/2
            trj_rs[..., i] = torch.clamp(trj_rs[..., i] + n_over_2, 0, self.im_size_os[i]-1)
        trj_rs = torch.round(trj_rs).type(torch.int32)

        return trj_rs

    def forward_FT_only(self,
                        img: torch.Tensor) -> torch.Tensor:
        
        # Consts
        d = len(self.im_size)
        
        # To torch
        img_torch = np_to_torch(img)

        # Oversampled FFT
        img_os = resize(img_torch, tuple(img.shape[:-d]) + self.im_size_os)
        ksp_os = fft(img_os, dim=tuple(range(-d, 0)))
        
        return ksp_os
    
    def forward_interp_only(self,
                            ksp_os: torch.Tensor,
                            trj: torch.Tensor) -> torch.Tensor:
        # consts
        trj_torch = np_to_torch(trj)
        d = trj.shape[-1]
        N = trj.shape[0]
        
        # Return k-space
        ksp = torch.zeros((*ksp_os.shape[:-d], *trj.shape[1:-1]), 
                          dtype=complex_dtype, device=ksp_os.device)
        for i in range(N):
            ksp[i] = multi_index(ksp_os[i], d, trj_torch[i].type(torch.int32))
        
        return ksp * self.grid_oversamp

    def adjoint_iFT_only(self, 
                         ksp_os: torch.Tensor) -> torch.Tensor:
        # Consts
        d = len(self.im_size)
        
        # iFFT and crop
        img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
        img = resize(img_os, tuple(img_os.shape[:-d]) + self.im_size)
        
        return img * self.grid_oversamp
    
    def adjoint_grid_only(self, 
                          ksp: torch.Tensor, 
                          trj: torch.Tensor) -> torch.Tensor:
        # To torch
        ksp_torch, trj_torch = np_to_torch(ksp, trj)
        
        # Consts
        d = trj.shape[-1]
        N = trj.shape[0]

        # Adjoint NUFFT
        ksp_os = torch.zeros((*ksp.shape[:-(trj.ndim - 2)], *self.im_size_os), 
                             dtype=complex_dtype, device=ksp_torch.device)
        for i in range(N):
            ksp_os[i] = multi_grid(ksp_torch[i], trj_torch[i].type(torch.int32), self.im_size_os)
            
        return ksp_os
    
    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = None):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz (unused here)

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        os_factor = self.grid_oversamp
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = gridded_nufft(im_size_os, grid_oversamp=1.0)

        return calc_toep_kernel_helper(nufft_os.adjoint, (trj).type(torch.int32), weights) * (os_factor ** len(self.im_size))

class lr_nufft(NUFFT):
    """
    Low Rank Nufft:
    e^{-j2\pi k \cdot r} = \sum_{l=1}^{L} b_l(r) * b_l(r).conj()
    """
    
    def __init__(self, 
                 im_size: tuple,
                 os_grid: Optional[float] = 1.0,
                 L: Optional[int] = 5):
        super().__init__(im_size)
        self.os_grid = os_grid
        self.padder = PadLast(tuple([round(i * os_grid) for i in self.im_size]), im_size)
        self.L = L

    def compute_basis_funcs(self,
                            matrix_size: tuple,
                            torch_dev: Optional[torch.device] = torch.device('cpu'),
                            use_toeplitz: Optional[bool] = True) -> dict:
        """
        Computes basis functions for fourier deviations from 
        a regular grid.
        
        Parameters:
        -----------
        matrix_size: tuple
            size of the matrix to compute the eigen-decomp on
        torch_dev : torch.device
            device to compute on
        use_toeplitz : bool
            use toeplitz for faster computation
        
        Returns:
        --------
        dict {
            'temporal_funcs': torch.Tensor <complex>,
                - has shape (L, *matrix_size)
            'spatial_funcs': torch.Tensor <complex>,
                - has shape (L, *matrix_size)
            'ks': torch.Tensor <float>
                - has shape matrix_size with valuse between -0.5 and 0.5
            'rs': torch.Tensor <float>
                - has shape matrix_size with valuse between -0.5 and 0.5
        }
        basis_funcs : torch.Tensor <complex>
            temporal basis functions with shape (L, *matrix_size)
        """
        # Define grids
        ks = gen_grd(matrix_size, balanced=True).to(torch_dev) / self.os_grid
        rs = gen_grd(matrix_size, balanced=True).to(torch_dev)

        # Use other NUFFT to compute SVD quickly
        sp_nufft = sigpy_nufft(matrix_size, oversamp=2.0)
        ks = -sp_nufft.rescale_trajectory(ks)
        nvox = torch.prod(torch.tensor(matrix_size)).item() * (self.os_grid ** len(matrix_size))

        # Define forward, adjoint, graham operators
        def A(x):
            return sp_nufft.adjoint(x[None,], ks[None,])[0] * nvox ** 0.5
        def AH(y):
            return sp_nufft(y[None,], ks[None,])[0] * nvox ** 0.5
        if use_toeplitz:
            kerns = sp_nufft.calc_teoplitz_kernels(ks[None])
            def AAH(y):
                return sp_nufft.normal_toeplitz(y[None,None,], kerns)[0,0] * nvox
        else:
            def AAH(y):
                return A(AH(y))
            
        # Compute SVD
        U, S, Vh = svd_power_method_tall(A=AH,
                                         AHA=AAH,
                                         niter=100,
                                         inp_dims=matrix_size,
                                         rank=self.L,
                                         device=torch_dev)
        temporal_funcs = rearrange(U * S, '... L -> L ...').conj()
        spatial_funcs = Vh.conj()

        # Clear mem
        gc.collect()
        with torch_dev:
            torch.cuda.empty_cache()
            
        # Return stuff
        dct = {
            'temporal_funcs': temporal_funcs,
            'spatial_funcs': spatial_funcs,
            'ks': ks,
            'rs': rs,
        }

        return dct

    @staticmethod
    def interp_basis_funcs(basis_funcs: torch.Tensor,
                           grid: torch.Tensor,
                           grid_new: torch.Tensor) -> torch.Tensor:
        """
        Interpolates basis funcitons.
        
        Parameters:
        -----------
        basis_funcs : torch.Tensor <complex>
            basis functions with shape (L, *matrix_size)
        grid : torch.Tensor <float>
            spatial grid with shape (*matrix_size, d)
        grid_new : torch.Tensor <float>
            output spatial grid with shape (..., d)
            
        Note:
        -----
        len(matrix_size) must be either 2 or 3, otherwise torch grid_sample doesn't work.
        
        Returns:
        --------
        basis_funcs_new : torch.Tensor <complex>
            interpolated basis functions with shape (L, ...)
        """
        # Consts
        L = basis_funcs.shape[0]
        matrix_size = basis_funcs.shape[1:]
        d = len(matrix_size)
        assert grid.shape[:-1] == matrix_size
        assert grid.shape[-1] == d
        assert grid_new.shape[-1] == d
        assert d == 2 or d == 3
        
        # Rescale
        scale = grid.abs().max()
        
        # Reshape 
        tup = (slice(None),) + (None,) * (d-1) + (slice(None),)
        grid_new_flt = rearrange(grid_new, '... d -> (...) d')[tup] / scale # N 1 (1) d
        grid_new_flt = grid_new_flt.flip(dims=[-1])
        
        # Interpolate
        align_corners = True
        mode = 'nearest'
        basis_funcs_flt_r = nn.functional.grid_sample(basis_funcs[None,].real, 
                                                      grid=grid_new_flt[None,],
                                                      mode=mode, 
                                                      align_corners=align_corners)[0]
        basis_funcs_flt_i = nn.functional.grid_sample(basis_funcs[None,].imag, 
                                                      grid=grid_new_flt[None,],
                                                      mode=mode, 
                                                      align_corners=align_corners)[0]
        basis_funcs_flt = (basis_funcs_flt_r + 1j * basis_funcs_flt_i).type(dtypes.complex_dtype)
        
        # Reshape
        basis_funcs_flt = basis_funcs_flt.squeeze().reshape((L, *grid_new.shape[:-1]))
        # basis_funcs_flt = basis_funcs_flt.squeeze().T.reshape((*grid_new.shape[:-1], L))
        # basis_funcs_flt = basis_funcs_flt.moveaxis(-1, 0)
        
        return basis_funcs_flt
         
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        # Consts
        os_size = self.padder.pad_im_size
        
        # Compute basis functions
        matrix_size = self.im_size
        # matrix_size = (50,) * trj.shape[-1]
        dct = self.compute_basis_funcs(matrix_size=matrix_size,
                                       torch_dev=trj.device,
                                       use_toeplitz=True)
        ks_eval = trj - (trj*self.os_grid).round()/self.os_grid
        rs_eval = gen_grd(self.im_size, balanced=True).to(trj.device)
        
        # Interpolate basis functions
        self.temporal_funcs = self.interp_basis_funcs(dct['temporal_funcs'], dct['ks'], ks_eval)
        self.spatial_funcs = self.interp_basis_funcs(dct['spatial_funcs'], dct['rs'], rs_eval)
            
        # Clamp each dimension
        trj_rs = trj * self.os_grid
        for i in range(trj_rs.shape[-1]):
            n_over_2 = os_size[i]/2
            trj_rs[..., i] = torch.clamp(trj_rs[..., i] + n_over_2, 0, os_size[i]-1)
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
        L = self.L

        # Multiplied by spatial functions
        empty_dims = img_torch.ndim - self.spatial_funcs.ndim
        tup = (None, slice(0, L)) + (None,) * empty_dims + (slice(None),) * (self.spatial_funcs.ndim - 1)
        img_torch = img_torch[:, None, ...] * self.spatial_funcs[tup]

        # Oversampled FFT
        img_os = self.padder.forward(img_torch)
        ksp_os = fft(img_os, dim=tuple(range(-d, 0)))
        
        # Return k-space
        ksp = torch.zeros((*img_torch.shape[:-d], *trj.shape[1:-1]), 
                          dtype=dtypes.complex_dtype, device=img_torch.device)
        for i in range(N):
            ksp[i] = multi_index(ksp_os[i], d, trj_torch[i].type(torch.int32))

        empty_dims = ksp.ndim - self.temporal_funcs.ndim - 1
        tup = (None, slice(0, L)) + (None,) * empty_dims + (slice(None),) * (self.temporal_funcs.ndim - 1)
        ksp = (ksp * self.temporal_funcs[tup]).sum(1)
        
        return ksp * self.os_grid
                    
    def adjoint(self, 
                ksp: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        """
        Adjoint non-uniform fourier transform

        Parameters:
        -----------
        ksp : torch.Tensor <complex>
            input k-space with shape (N, *ksp_batch, *trj_batch)
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        img : torch.Tensor <complex>
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
        L = self.L
        os_size = self.padder.pad_im_size

        # Multiply by temporal functions
        empty_dims = ksp.ndim - self.temporal_funcs.ndim
        tup = (None, slice(0, L)) + (None,) * empty_dims + (slice(None),) * (self.temporal_funcs.ndim - 1)
        ksp_torch = ksp_torch[:, None, ...] * self.temporal_funcs.conj()[tup]

        # Adjoint NUFFT
        ksp_os = torch.zeros((*ksp_torch.shape[:-(trj.ndim - 2)], *os_size), 
                             dtype=dtypes.complex_dtype, device=ksp_torch.device)
        for i in range(N):
            ksp_os[i] = multi_grid(ksp_torch[i], trj_torch[i].type(torch.int32), os_size)
        img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
        img = self.padder.adjoint(img_os)

        empty_dims = img.ndim - self.spatial_funcs.ndim - 1
        tup = (None, slice(0, L)) + (None,) * empty_dims + (slice(None),) * (self.spatial_funcs.ndim - 1)
        img = (img * self.spatial_funcs.conj()[tup]).sum(1)

        return img * self.os_grid

class svd_nufft(NUFFT):
    """
    Models deviations from regular grid with SVD low rank model
    """
    # TODO impliment batching?
    def __init__(self,
                 im_size: tuple,
                 grid_oversamp: Optional[float] = 1.0,
                 n_svd: Optional[int] = 5,
                 n_batch_size: Optional[int] = None,
                 svd_mx_size: Optional[int] = None):
        super().__init__(im_size)
        os_size = tuple([round(i * grid_oversamp) for i in self.im_size])
        self.grid_oversamp = grid_oversamp
        self.padder = PadLast(os_size, im_size)
        self.n_svd = n_svd
        self.n_batch_size = n_svd if n_batch_size is None else n_batch_size
        self.svd_mx_size = (50,) * len(im_size) if svd_mx_size is None else svd_mx_size
        assert len(self.svd_mx_size) == len(im_size), "svd_mx_size must match im_size in length"
        
        
    @staticmethod
    def spatial_interp(spatial_input: torch.Tensor, 
                       coords: torch.Tensor, 
                       order: Optional[int] = 3, 
                       mode: Optional[str] = 'nearest') -> torch.Tensor:
        """
        Perform polynomial interpolation on the spatial dimensions of a tensor
        at specified coordinates.
        
        Parameters:
        -----------
        spatial_input : torch.Tensor
            Input tensor of shape (N, d0, d1, ..., d_{K-1}).
        coords : torch.Tensor
            target coordinates with shape (*crds_size, K)
        order : int, optional
            The order of the spline interpolation (default is cubic, order=3).
        mode : str, optional
            How to handle points outside the boundaries (default 'nearest').
        
        Returns:
        --------
        out : torch.Tensor
            An array of interpolated values with shape (N, *crds_size). That is, for each
            batch element, the tensor is evaluated at the provided coordinates.
        """
        # Convert to cupy
        torch_dev = spatial_input.device
        spatial_input_cp = torch_to_np(spatial_input)
        crds_flt_cp = torch_to_np(coords).reshape((-1, coords.shape[-1]))  # Flatten for map_coordinates
        dev = sp.get_device(spatial_input_cp)
        xp = dev.xp
        
        # Consts
        with dev:
            M, K = crds_flt_cp.shape
            N = spatial_input_cp.shape[0]
            out = xp.empty((N, M), dtype=spatial_input_cp.dtype)
            assert spatial_input_cp.ndim == K + 1, "Input tensor must have K spatial dimensions."
            
            # Loop over batch elements.
            for i in range(N):
                interp_vals = map_coordinates(
                    spatial_input_cp[i], crds_flt_cp.T, order=order, mode=mode
                )
                out[i] = interp_vals
        
        # Convert back to torch
        out = np_to_torch(out)
        out = out.reshape((N, *coords.shape[:-1]))  # Reshape to (N, *crds_size)
        
        return out

    def compute_svd_funcs(self,
                          trj: torch.Tensor,
                          batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Computes the SVD functions for fourier deviations
        """
        # Consts
        im_size = self.im_size
        os_grid = self.grid_oversamp
        torch_dev = trj.device
        
        # Spatial crds
        svd_size = self.svd_mx_size
        rs = gen_grd(svd_size).to(torch_dev)
        rs = rs.reshape((-1, len(svd_size)))
        
        # Temporal kspace crds
        trj_dev = trj - (trj * os_grid).round() / os_grid
        # ks = trj_dev.reshape((-1, len(im_size)))
        ks = rs.clone()
        
        # Build matrix
        n, _ = rs.shape
        m, _ = ks.shape
        mx = torch.zeros((m, n), dtype=complex_dtype, device=torch_dev)
        if batch_size is None:
            batch_size = m
        for m1 in range(0, m, batch_size):
            m2 = min(m1 + batch_size, m)
            phz = ks[m1:m2] @ rs.T # (m1:m2, n)
            mx[m1:m2, :] = torch.exp(-2j * torch.pi * phz)
        MHM = mx.H @ mx
        
        # Linear operators
        def forward(x):
            # x is (k, *svd_size)
            k = x.shape[0]
            x_vec = x.reshape((k, -1)).T
            out = mx @ x_vec
            return out.T.reshape((k, *svd_size))
        def gram(x):
            # x is (k, *svd_size)
            k = x.shape[0]
            x_vec = x.reshape((k, -1)).T
            out = MHM @ x_vec
            return out.T.reshape((k, *svd_size))
            
        # eigen-decomp
        x0 = torch.ones(svd_size, dtype=complex_dtype, device=torch_dev)
        evecs, _ = eigen_decomp_operator(gram, x0, num_eigen=self.n_svd, num_iter=100, lobpcg=True)
        temp_evecs = forward(evecs)
        
        # Cleanup memory
        del mx, MHM
        gc.collect()
        with torch_dev:
            torch.cuda.empty_cache()
        
        # Interpolate spatial functions
        kwargs = {'order': 3, 'mode': 'nearest'}
        svd_size_tensor = torch.tensor(svd_size).to(torch_dev)
        spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * svd_size_tensor
        spatial_funcs = self.spatial_interp(evecs, spatial_crds, **kwargs)
        
        # Interpolate temporal functions
        temporal_crds = (0.5 + trj_dev * os_grid) * svd_size_tensor
        temporal_funcs = self.spatial_interp(temp_evecs, temporal_crds, **kwargs)
        
        
        return temporal_funcs, spatial_funcs.conj()
    
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Compute basis functions
        self.temporal_funcs, self.spatial_funcs = self.compute_svd_funcs(trj)
        
        # Clamp each dimension
        trj_rs = trj * self.grid_oversamp
        grid_os_size = self.padder.pad_im_size
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
        n_svd = self.n_svd

        # Multiplied by spatial functions
        empty_dims = img_torch.ndim - self.spatial_funcs.ndim
        tup = (None, slice(0, n_svd)) + (None,) * empty_dims + (slice(None),) * (self.spatial_funcs.ndim - 1)
        img_torch = img_torch[:, None, ...] * self.spatial_funcs[tup]

        # Oversampled FFT
        img_os = self.padder.forward(img_torch)
        ksp_os = fft(img_os, dim=tuple(range(-d, 0)))
        
        # Return k-space
        ksp = torch.zeros((*img_torch.shape[:-d], *trj.shape[1:-1]), 
                          dtype=dtypes.complex_dtype, device=img_torch.device)
        for i in range(N):
            ksp[i] = multi_index(ksp_os[i], d, trj_torch[i].type(torch.int32))

        empty_dims = ksp.ndim - self.temporal_funcs.ndim - 1
        tup = (None, slice(0, n_svd)) + (None,) * empty_dims + (slice(None),) * (self.temporal_funcs.ndim - 1)
        ksp = (ksp * self.temporal_funcs[tup]).sum(1)
        
        return ksp * self.grid_oversamp
                    
    def adjoint(self, 
                ksp: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        """
        Adjoint non-uniform fourier transform

        Parameters:
        -----------
        ksp : torch.Tensor <complex>
            input k-space with shape (N, *ksp_batch, *trj_batch)
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        img : torch.Tensor <complex>
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
        n_svd = self.n_svd
        grid_os_size = self.padder.pad_im_size

        # Multiply by temporal functions
        empty_dims = ksp.ndim - self.temporal_funcs.ndim
        tup = (None, slice(0, n_svd)) + (None,) * empty_dims + (slice(None),) * (self.temporal_funcs.ndim - 1)
        ksp_torch = ksp_torch[:, None, ...] * self.temporal_funcs.conj()[tup]

        # Adjoint NUFFT
        ksp_os = torch.zeros((*ksp_torch.shape[:-(trj.ndim - 2)], *grid_os_size), 
                             dtype=dtypes.complex_dtype, device=ksp_torch.device)
        for i in range(N):
            ksp_os[i] = multi_grid(ksp_torch[i], trj_torch[i].type(torch.int32), grid_os_size)
        img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
        img = self.padder.adjoint(img_os)

        empty_dims = img.ndim - self.spatial_funcs.ndim - 1
        tup = (None, slice(0, n_svd)) + (None,) * empty_dims + (slice(None),) * (self.spatial_funcs.ndim - 1)
        img = (img * self.spatial_funcs.conj()[tup]).sum(1)

        return img * self.grid_oversamp
    
class chebyshev_nufft(NUFFT):
    """
    NUFFT based on https://arxiv.org/pdf/1701.04492
    Lemma A.3 here is most relevent: https://pi.math.cornell.edu/~ajt/papers/thesis.pdf

    Uses chebyshev expansion on grid deviations as an analytical low rank model.
    """

    def __init__(self,
                 im_size: tuple,
                 n_cheby_per_dim: Optional[int] = 5,
                 grid_oversamp: Optional[float] = 1.0,
                 n_batch_size: Optional[int] = None):
        super().__init__(im_size)
        grd_os_size = tuple([round(i * grid_oversamp) for i in self.im_size])
        self.grid_oversamp = grid_oversamp
        self.grog_padder = PadLast(grd_os_size, im_size)
        self.gamma = 1 / (2 * grid_oversamp)
        self.n_cheby_per_dim = n_cheby_per_dim
        self.n_batch_size = n_cheby_per_dim ** len(im_size) if n_batch_size is None else n_batch_size
    
    @staticmethod
    def cheby_weights(p, r, gamma):
        if (abs(p - r) % 2) == 0:
            scale = 4 * (1j ** r)
            J1 = jv((r+p)/2, -gamma * torch.pi/2)
            J2 = jv((r-p)/2, -gamma * torch.pi/2)
            return scale * J1 * J2
        else:
            return 0
        
    def compute_spatial_funcs(self,
                              im_size: tuple) -> torch.Tensor:

        # Consts
        n_cheby = self.n_cheby_per_dim
        gamma = self.gamma
        d = len(im_size)

        # Chebyshev polynomials
        T = lambda x, n : torch.cos(n * torch.arccos(x))

        # Make image basis functions
        grd = gen_grd(im_size)
        b = torch.zeros((d, n_cheby, *im_size),dtype=dtypes.complex_dtype)
        for i in range(d):
            r = grd[..., i]
            for lp in range(n_cheby):
                if lp == 0:
                    scale_p = 1/2
                else:
                    scale_p = 1
                for lq in range(n_cheby):
                    if lq == 0:
                        scale_q = 1/2
                    else:
                        scale_q = 1
                    scale = scale_q * scale_p
                    # scale = scale_q
                    b[i, lp] += T(2 * r, lq) * self.cheby_weights(lp, lq, gamma) * scale
        
        return b

    def compute_temporal_funcs(self,
                               trj: torch.Tensor) -> torch.Tensor:
        
        # Consts
        trj_size = trj.shape[:-1]
        n_cheby = self.n_cheby_per_dim
        grid_oversamp = self.grid_oversamp
        gamma = self.gamma
        d = trj.shape[-1]

        # Grid deviations
        trj_dev = trj - torch.round(trj * grid_oversamp) / grid_oversamp

        # Chebyshev polynomials
        T = lambda x, n : torch.cos(n * torch.arccos(x))

        # Make temporal basis functions
        h = torch.zeros((d, n_cheby, *trj_size),dtype=dtypes.complex_dtype)
        for i in range(d):
            k = trj_dev[..., i]
            for lp in range(n_cheby):
                h[i, lp] = T(k / gamma, lp)
        
        return h
    
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Clamp each dimension
        trj_rs = trj * self.grid_oversamp
        grid_os_size = self.grog_padder.pad_im_size
        for i in range(trj_rs.shape[-1]):
            n_over_2 = grid_os_size[i]/2
            trj_rs[..., i] = torch.clamp(trj_rs[..., i] + n_over_2, 0, grid_os_size[i]-1)
        trj_rs = torch.round(trj_rs).type(torch.int32)

        # Compute basis functions
        h = self.compute_temporal_funcs(trj)
        b = self.compute_spatial_funcs(self.im_size)
        d = trj.shape[-1]
        if d == 1:
            h = h[0]
            b = b[0]
        elif d == 2:
            h = einsum(h[0], h[1], 'L1 ..., L2 ... -> L1 L2 ...').reshape((-1, *trj.shape[:-1]))
            b = einsum(b[0], b[1], 'L1 ..., L2 ... -> L1 L2 ...').reshape((-1, *self.im_size))
        elif d == 3:
            h = einsum(h[0], h[1], h[2], 'L1 ..., L2 ..., L3 ... -> L1 L2 L3 ...').reshape((-1, *trj.shape[:-1]))
            b = einsum(b[0], b[1], b[2], 'L1 ..., L2 ..., L3 ... -> L1 L2 L3 ...').reshape((-1, *self.im_size))
        self.h = h.to(trj.device)
        self.b = b.to(trj.device)

        return trj_rs

    def forward(self, 
                img: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        
        # To torch
        img_torch, trj_torch = np_to_torch(img, trj)

        # Consts
        d = trj.shape[-1]
        N = trj.shape[0]
        L = self.h.shape[0]

        # Return k-space
        ksp = torch.zeros((*img.shape[:-d], *trj.shape[1:-1]), 
                          dtype=dtypes.complex_dtype, device=img_torch.device)

        # Batch over basis functions
        for l1, l2 in batch_iterator(L, self.n_batch_size):

            # Multiply by spatial terms
            img_batches = img_torch.unsqueeze(-d-1) * self.b[l1:l2]

            # Oversampled FFT
            img_os = self.grog_padder.forward(img_batches)
            ksp_os = fft(img_os, dim=tuple(range(-d, 0)))
            
            for i in range(N):
                # Index correct points
                ksp_i = multi_index(ksp_os[i], d, trj_torch[i].type(torch.int32))

                # Multiply by temporal terms
                ksp[i] = (ksp_i * self.h[l1:l2]).sum(-self.h.ndim)
        
        return ksp * self.grid_oversamp
                    
    def adjoint(self, 
                ksp: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        """
        Adjoint non-uniform fourier transform

        Parameters:
        -----------
        ksp : torch.Tensor <complex>
            input k-space with shape (N, *ksp_batch, *trj_batch)
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        
        Returns:
        --------
        img : torch.Tensor <complex>
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
        L = self.h.shape[0]
        tb = len(trj.shape) - 2
        grid_os_size = self.grog_padder.pad_im_size

        # Adjoint NUFFT
        img = torch.zeros((*ksp.shape[:-tb], *self.im_size), dtype=dtypes.complex_dtype, device=ksp_torch.device)
        for l1, l2 in batch_iterator(L, self.n_batch_size):
            ksp_os = torch.zeros((*ksp.shape[:-(trj.ndim - 2)], (l2-l1), *grid_os_size), 
                                  dtype=dtypes.complex_dtype, device=ksp_torch.device)
            for i in range(N):
                # Multiply by temporal terms
                ksp_os[i] = multi_grid(ksp_torch[i].unsqueeze(-tb-1) * self.h[l1:l2].conj(), 
                                    trj_torch[i].type(torch.int32), grid_os_size)
            img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
            img_os_crp = self.grog_padder.adjoint(img_os)

            # Multiply by spatial terms
            img += (img_os_crp * self.b[l1:l2].conj()).sum(-self.b.ndim)

        return img * self.grid_oversamp
   