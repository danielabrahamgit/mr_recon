import torch
import numpy as np
import sigpy as sp
import torch.nn as nn

from math import ceil
from mr_recon.utils import np_to_torch, torch_to_np, resize, gen_grd
from mr_recon.dtypes import np_complex_dtype, complex_dtype
from mr_recon.fourier.common import fft, ifft, NUFFT, calc_toep_kernel_helper
from typing import Optional

from sigpy.fourier import (
    _get_oversamp_shape, 
    _scale_coord)
   
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
            if (((width / oversamp) * (oversamp - 0.5))**2 - 0.8) < 0:
                if width != 1:
                    print('WARNING: beta is set to 1.0')
                self.beta = 1  
            else:
                self.beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
        else:
            self.beta = beta
    
    def optimal_beta(self,
                     nbetas: Optional[int] = 100,
                     num_sigs: Optional[int] = 1000,
                     torch_dev: Optional[torch.device] = torch.device('cpu')) -> float:
        """
        Returns the optimal beta for this NUFFT with 
        
        Returns:
        --------
        beta : float
            optimal beta for the NUFFT
        """
        # Discretize over kdevs, rs
        N = max(self.im_size)
        kdevs = torch.linspace(-0.5, 0.5, N, device=torch_dev) / self.oversamp
        betas = torch.linspace(0.5, 20, nbetas, device=torch_dev)
        
        # Make a few random 1d signals
        rnd_sigs = torch.randn((num_sigs, N), dtype=complex_dtype, device=torch_dev)
        
        # Ground truth values
        rs = gen_grd((N,)).to(torch_dev)[:, 0]
        gt = torch.exp(-2j * np.pi * (kdevs[:, None] @ rs[None, :]))
        vals_gt = torch.zeros((num_sigs, N), dtype=complex_dtype, device=torch_dev)
        for i in range(num_sigs):
            vals_gt[i] = gt @ rnd_sigs[i]
        
        # See which beta is closest to the ground truth
        errs = []
        for beta in betas:
            
            # Compare beta-nufft to ground truth
            nft = sigpy_nufft((N,), self.oversamp, self.width, beta=beta)
            vals = nft.forward(rnd_sigs[None,], kdevs[None, :, None])[0]
            scale = (vals.conj() * vals_gt).sum() / (vals.conj() * vals).sum()
            vals = vals * scale
            err = (vals - vals_gt).norm()
            errs.append(err)
        
        # Return best Beta
        errs = torch.stack(errs, dim=0)
        imin = errs.argmin(dim=0)
  
        return betas[imin].item()
    
    def _apodization_func_kb(self, 
                             x: torch.Tensor,
                             beta: float,
                             width: float) -> torch.Tensor:
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
        eps = 1e-12
        arg = (beta**2 - (np.pi * width * x) ** 2)
        apod_pos = arg.clamp(min=0).sqrt()
        apod_pos /= torch.sinh(apod_pos) + eps
        apod_neg = (-arg.clamp(max=0)).sqrt()
        apod_neg /= torch.sin(apod_neg) + eps
        return apod_pos + apod_neg
    
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
            img *= self._apodization_func_kb(crds / ceil(self.oversamp * img.shape[i]),
                                             beta=self.beta,
                                             width=self.width)[tup]
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
            ksp_ret = dev.xp.zeros((N, *img_shape[1:-ndim], *trj_cp.shape[1:-1]), dtype=np_complex_dtype)
            for i in range(N):
                ksp_ret[i] = sp.interp.interpolate(
                        ksp_os_cp[i], trj_cp[i], kernel='kaiser_bessel', width=width, param=beta)
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
            output = dev.xp.zeros(os_shape, dtype=np_complex_dtype)
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

class _NUFFTLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img: torch.Tensor, trj: torch.Tensor, nufft: "NUFFT"):
        """
        img: (N, *img_batch, *im_size) complex
        trj: (N, *trj_batch, d) float  (treated as constant)
        nufft: instance of your NUFFT subclass
        """
        # Save only what we need for backward
        ctx.nufft = nufft
        ctx.save_for_backward(trj)
        # Forward NUFFT
        return nufft.forward(img, trj)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        grad_out: ∂L/∂y (same shape as forward output)
        returns: ∂L/∂img, None, None
        """
        (trj,) = ctx.saved_tensors
        nufft = ctx.nufft

        # VJP for a linear complex operator: A^H @ grad_out
        grad_img = nufft.adjoint(grad_out, trj)

        # No grads for trj or nufft handle
        return grad_img, None, None


class nufft_wrapper(nn.Module):
    """
    Autograd-friendly NUFFT wrapper with gradients only w.r.t. the image.
    """
    def __init__(self, nufft: "NUFFT"):
        super().__init__()
        self.nufft = nufft

    def forward(self, img: torch.Tensor, trj: torch.Tensor) -> torch.Tensor:
        # Ensure we don't waste compute tracking trj
        if trj.requires_grad:
            trj = trj.detach()
        return _NUFFTLinearFn.apply(img, trj, self.nufft)
