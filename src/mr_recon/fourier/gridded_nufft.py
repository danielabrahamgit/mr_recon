import torch

from .common import NUFFT, fft, ifft, calc_toep_kernel_helper
from typing import Optional
from mr_recon.dtypes import complex_dtype
from mr_recon.indexing import (
    multi_grid,
    multi_index
)
from mr_recon.utils import np_to_torch,resize

  
class gridded_nufft(NUFFT):

    def __init__(self,
                 im_size: tuple,
                 oversamp: Optional[float] = 1.0,):
        super().__init__(im_size)
        self.im_size_os = tuple([round(i * oversamp) for i in self.im_size])
        self.oversamp = oversamp
    
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Clamp each dimension
        trj_rs = trj * self.oversamp
        for i in range(trj_rs.shape[-1]):
            n_over_2 = self.im_size_os[i]/2
            trj_rs[..., i] = (trj_rs[..., i] + n_over_2).round() % self.im_size_os[i] # $25 to Yonatan
            # trj_rs[..., i] = torch.clamp(trj_rs[..., i] + n_over_2, 0, self.im_size_os[i]-1)
        trj_rs = trj_rs.type(torch.int32)

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
        
        return ksp * (self.oversamp ** (d/2))

    def adjoint_iFT_only(self, 
                         ksp_os: torch.Tensor) -> torch.Tensor:
        # Consts
        d = len(self.im_size)
        
        # iFFT and crop
        img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
        img = resize(img_os, tuple(img_os.shape[:-d]) + self.im_size)
        
        return img
    
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
            
        return ksp_os * (self.oversamp ** (d/2))
    
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
        os_factor = self.oversamp
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = gridded_nufft(im_size_os, oversamp=1.0)

        return calc_toep_kernel_helper(nufft_os.adjoint, (trj).type(torch.int32), weights) * (os_factor ** len(self.im_size))
