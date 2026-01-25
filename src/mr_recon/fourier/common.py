import torch
import sigpy as sp
import torch.nn as nn
import torch.fft as fft_torch


from typing import Optional
from mr_recon.dtypes import complex_dtype, real_dtype
from mr_recon._func.pad import PadLast
from mr_recon.utils import (
    torch_to_np, 
    np_to_torch,
    batch_iterator)

def fft(x: torch.Tensor, 
        dim: Optional[tuple] = None, 
        oshape: Optional[tuple] = None, 
        norm: Optional[str] = 'ortho') -> torch.Tensor:
    """
    Matches Sigpy's fft, but in torch.
    
    Args
    ----
    x : torch.Tensor
        input tensor
    dim : tuple
        dimensions to perform fft on
    oshape : tuple
        output shape, if resizing is needed
    norm : str
        normalization type, default is 'ortho'
        
    Returns
    -------
    torch.Tensor
        output tensor with same shape as input or oshape if provided
    """

    if oshape is not None:
        x_cp = torch_to_np(x)
        dev = sp.get_device(x_cp)
        with dev:
            x = np_to_torch(sp.resize(x_cp, oshape))
    x = fft_torch.ifftshift(x, dim=dim)
    x = fft_torch.fftn(x, dim=dim, norm=norm)
    x = fft_torch.fftshift(x, dim=dim)
    return x

def ifft(x: torch.Tensor, 
         dim: Optional[tuple] = None, 
         oshape: Optional[tuple] = None, 
         norm: Optional[str] = 'ortho') -> torch.Tensor:
    """
    Matches Sigpy's ifft, but in torch.
    
    Args
    ----
    x : torch.Tensor
        input tensor
    dim : tuple
        dimensions to perform fft on
    oshape : tuple
        output shape, if resizing is needed
    norm : str
        normalization type, default is 'ortho'
        
    Returns
    -------
    torch.Tensor
        output tensor with same shape as input or oshape if provided
    """

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
        weights = torch.ones(trj.shape[:-1], dtype=real_dtype, device=torch_dev)
    else:
        assert weights.device == torch_dev
    trj_batch = trj.shape[1:-1]
    if trj.shape[0] == 1 and weights.shape[0] != 1:
        trj = trj.expand((weights.shape[0], *trj.shape[1:]))
    N = trj.shape[0]
    d = trj.shape[-1]
    
    # Get toeplitz kernel via adjoint nufft on 1s ksp
    ksp = torch.ones((N, 1, *trj_batch), device=torch_dev, dtype=complex_dtype)
    ksp_weighted = ksp * weights[:, None, ...]
    img = nufft_adj_os(ksp_weighted, trj)[:, 0, ...] # (N, *im_size_os)        

    # FFT
    toeplitz_kernels = fft(img, dim=tuple(range(-d, 0)))

    return toeplitz_kernels

class NUFFT(nn.Module):
    """
    Forward NUFFT is defined as:
    $$y(\mathbf{k}) = \int_\mathbf{r} x(\mathbf{r}) e^{-j2\pi \mathbf{k} \cdot \mathbf{r}} d\mathbf{r}$$ 
    
    Please $t_0$ work.

    where $$k_i \in [-N_i/2, N_i/2]$$ and `im_size[i]` $$= N_i$$
    and $$r_i \in [-1/2, 1/2]$$.
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
