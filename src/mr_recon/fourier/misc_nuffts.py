import gc
import torch
import numpy as np
import sigpy as sp
import cufinufft

from sigpy.fourier import _get_oversamp_shape

from .common import NUFFT, fft, ifft, calc_toep_kernel_helper

from typing import Optional
from einops import einsum
from scipy.special import jv
from cupyx.scipy.ndimage import map_coordinates
from math import ceil
from mr_recon.dtypes import complex_dtype, real_dtype
from mr_recon._func.pad import PadLast
from mr_recon.algs import eigen_decomp_operator
from mr_recon._func.indexing import (
    multi_grid,
    multi_index
)
from mr_recon.utils import (
    gen_grd,
    torch_to_np, 
    np_to_torch,
    batch_iterator,
    resize)

class cufi_nufft(NUFFT):
    
    def __init__(self,
                 im_size: tuple,
                 eps: Optional[float] = 1e-4):
        """
        Uses the cufi-nufft library to perform the NUFFT.
        
        Args:
        -----
        im_size : tuple
            The size of the image to be transformed.
        eps : float, optional
            The epsilon value for the cufi-nufft library, default is 1e-4.
        """
        super().__init__(im_size)
        self.eps = eps
        
        if len(im_size) == 1:
            self.frw = cufinufft.nufft1d2
            self.adj = cufinufft.nufft1d1
        elif len(im_size) == 2:
            self.frw = cufinufft.nufft2d2
            self.adj = cufinufft.nufft2d1
        elif len(im_size) == 3:
            self.frw = cufinufft.nufft3d2
            self.adj = cufinufft.nufft3d1

    def forward(self,
                img: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        N = trj.shape[0]
        d = trj.shape[-1]
        img_batch_size = img.shape[1:-len(self.im_size)]
        im_size = self.im_size
        trj_size = trj.shape[:-1]
        trj_flt = trj.reshape((N, -1, d))
        img_flt = img.reshape((N, -1, *im_size))
        
        ksp = torch.zeros((N, *img_batch_size, *trj.shape[1:-1]), 
                          dtype=complex_dtype, device=img.device)
        
        for n in range(N):
            trj_cp = torch_to_np(trj_flt[n])
            img_cp = torch_to_np(img_flt[n])
            ksp_n = np_to_torch(self.frw(*trj_cp.T, 
                                         img_cp, 
                                         eps=self.eps,
                                         isign=-1))
            ksp[n] = ksp_n.reshape((*img.shape[:-d], *trj_size))
        return ksp / np.prod(im_size) ** 0.5
    
    def adjoint(self,
                ksp: torch.Tensor,
                trj: torch.Tensor) -> torch.Tensor:
        N = trj.shape[0]
        d = trj.shape[-1]
        im_size = self.im_size
        trj_size = trj.shape[:-1]
        ksp_size = ksp.shape[:-len(trj_size)]
        trj_flt = trj.reshape((N, -1, d))
        ksp_flt = ksp.reshape((N, -1, trj_flt.shape[1]))
        
        img = torch.zeros((N, *ksp.shape[1:-len(trj.shape[1:-1])], *self.im_size), 
                          dtype=complex_dtype, device=ksp.device) 
        for n in range(N):
            trj_cp = torch_to_np(trj_flt[n])
            ksp_cp = torch_to_np(ksp_flt[n])
            img_n = np_to_torch(self.adj(*trj_cp.T,
                                         ksp_cp, 
                                         eps=self.eps,
                                         isign=1,
                                         n_modes=im_size))
            img[n] = img_n.reshape((*ksp_size, *im_size))
        
        return img / np.prod(im_size) ** 0.5

    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Rescale to -pi, pi
        im_size_arr = torch.tensor(self.im_size).to(trj.device)
        tup = (None,) * (trj.ndim - 1) + (slice(None),)
        trj_rs = torch.pi * trj / (im_size_arr[tup] / 2)

        return trj_rs.contiguous()
  
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
            trj_rs[..., i] = (trj_rs[..., i] + n_over_2).round() % grid_os_size[i] # $25 to Yonatan
            # trj_rs[..., i] = torch.clamp(trj_rs[..., i] + n_over_2, 0, grid_os_size[i]-1)
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
                          dtype=complex_dtype, device=img_torch.device)
        for i in range(N):
            ksp[i] = multi_index(ksp_os[i], d, trj_torch[i].type(torch.int32))

        # Multiply by temporal functions
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
                             dtype=complex_dtype, device=ksp_torch.device)
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
        b = torch.zeros((d, n_cheby, *im_size),dtype=complex_dtype)
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
        h = torch.zeros((d, n_cheby, *trj_size),dtype=complex_dtype)
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
                          dtype=complex_dtype, device=img_torch.device)

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
        img = torch.zeros((*ksp.shape[:-tb], *self.im_size), dtype=complex_dtype, device=ksp_torch.device)
        for l1, l2 in batch_iterator(L, self.n_batch_size):
            ksp_os = torch.zeros((*ksp.shape[:-(trj.ndim - 2)], (l2-l1), *grid_os_size), 
                                  dtype=complex_dtype, device=ksp_torch.device)
            for i in range(N):
                # Multiply by temporal terms
                ksp_os[i] = multi_grid(ksp_torch[i].unsqueeze(-tb-1) * self.h[l1:l2].conj(), 
                                    trj_torch[i].type(torch.int32), grid_os_size)
            img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
            img_os_crp = self.grog_padder.adjoint(img_os)

            # Multiply by spatial terms
            img += (img_os_crp * self.b[l1:l2].conj()).sum(-self.b.ndim)

        return img * self.grid_oversamp
   