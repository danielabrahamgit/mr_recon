import torch
import numpy as np

from mr_recon.fourier import NUFFT
from mr_recon.utils import gen_grd
from mr_recon.dtypes import complex_dtype
from mr_recon.fourier.common import calc_toep_kernel_helper
from typing import Optional
from tqdm import tqdm

class matrix_nufft(NUFFT):
    
    def __init__(self, 
                 im_size: tuple,
                 spatial_batch_size: Optional[int] = None,):
        self.im_size = im_size
        self.batch_size = np.prod(im_size) if spatial_batch_size is None else spatial_batch_size
        super().__init__(im_size)

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
        # Consts
        torch_dev = img.device
        im_size = self.im_size
        trj_size = trj.shape[1:-1]
        img_batch = img.shape[1:-len(im_size)]
        N = img.shape[0]
        R = np.prod(im_size)
        T = np.prod(trj_size)
        d = len(im_size)
        
        # Flatten everything
        img_flt = img.reshape((N, -1, R))
        trj_flt = trj.reshape((N, T, d))
        rs = gen_grd(im_size).to(torch_dev).reshape((-1, d)) 
        
        # Return this
        ksp = torch.zeros((N, *img_batch, *trj_size), dtype=complex_dtype, device=torch_dev)
        
        for n in range(N):
            # Create encoding matrix over batch
            for r1 in range(0, R, self.batch_size):
                r2 = min(r1 + self.batch_size, R)
                phz = rs[r1:r2] @ trj_flt[n].T # (R T)
                enc = torch.exp(-2j * np.pi * phz) # (R T)
                ksp_flt = img_flt[n, :, r1:r2] @ enc # (-1 T)
                ksp[n] += ksp_flt.reshape((*img_batch, *trj_size)) / np.sqrt(R)
                
        return ksp
    
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
        # Consts
        torch_dev = ksp.device
        im_size = self.im_size
        trj_size = trj.shape[1:-1]
        ksp_batch = ksp.shape[1:-len(trj_size)]
        N = ksp.shape[0]
        R = np.prod(im_size)
        T = np.prod(trj_size)
        d = len(im_size)
        
        # Flatten everything
        ksp_flt = ksp.reshape((N, -1, T))
        trj_flt = trj.reshape((N, T, d))
        rs = gen_grd(im_size).to(torch_dev).reshape((-1, d)) 
        
        # Return this
        img = torch.zeros((N, *ksp_batch, R), dtype=complex_dtype, device=torch_dev)
        
        for n in range(N):
            # Create encoding matrix over batch
            for r1 in tqdm(range(0, R, self.batch_size)):
                r2 = min(r1 + self.batch_size, R)
                phz = trj_flt[n] @ rs[r1:r2].T # (T R)
                enc = torch.exp(2j * np.pi * phz) # (T R)
                ksp_flt_n = ksp_flt[n, :, :] @ enc # (-1 R)
                img[n, ..., r1:r2] = ksp_flt_n.reshape((*ksp_batch, (r2-r1))) / np.sqrt(R)
        
        img = img.reshape((N, *ksp_batch, *im_size))
                
        return img
        
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
        nufft_os = matrix_nufft(im_size=im_size_os)

        return calc_toep_kernel_helper(nufft_os.adjoint, trj * os_factor, weights) * (os_factor ** len(self.im_size))
     