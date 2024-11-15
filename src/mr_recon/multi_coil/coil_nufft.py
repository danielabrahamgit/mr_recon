import torch
import gc
import torch.nn as nn

from mr_recon.dtypes import complex_dtype
from mr_recon.indexing import multi_index, multi_grid
from mr_recon.pad import PadLast
from mr_recon.utils import gen_grd, np_to_torch, batch_iterator
from mr_recon.algs import svd_power_method_tall
from mr_recon.spatial import spatial_resize

from mr_recon.fourier import NUFFT, sigpy_nufft, gridded_nufft, fft, ifft
from einops import rearrange
from typing import Optional


class coil_nufft(NUFFT):
    """
    Low Rank Nufft with coils:
    vec{C}(r) * e^{-j2\pi k \cdot r} = \sum_{l=1}^{L} b_l(r) * vec{h}_l(k)
    """
    
    def __init__(self, 
                 mps: torch.Tensor,
                 os_grid: Optional[float] = 1.0,
                 L: Optional[int] = 5,
                 coil_batch_size: Optional[int] = 1,
                 store_svd_interps: Optional[bool] = False,
                 spatial_interp_kind: Optional[str] = 'bicubic'):
        """
        Parameters:
        -----------
        mps : torch.Tensor <complex64>
            multi-coil sensitivity maps with shape (C, *im_size)
        os_grid : float
            oversampling grid factor (usually between 1.0 and 2.0)
        L : int
            number of basis functions to use
        store_svd_interps : bool
            if True, stores the temporal/spatial SVD interpolation matrices (costs memory)
            if False, interpolates on the fly (costs computation)
        spatial_interp_kind : str
            interpolation kind for spatial interpolation
            'nearest', 'bilinear', 'bicubic'
        """
        im_size = mps.shape[1:]
        super().__init__(im_size)
        self.os_grid = os_grid
        self.gnufft = gridded_nufft(im_size, grid_oversamp=os_grid)
        self.L = L
        self.store_svd_interps = store_svd_interps
        self.coil_batch_size = coil_batch_size
        self.spatial_interp_kind = spatial_interp_kind
        
        # Compute basis functions
        # matrix_size = im_size
        matrix_size = (50,) * len(im_size)
        basis_dict = self.compute_basis_funcs(mps,
                                              matrix_size=matrix_size,
                                              torch_dev=mps.device,
                                              use_toeplitz=True)
        self.basis_dict = basis_dict

    def compute_basis_funcs(self,
                            mps: torch.Tensor,
                            matrix_size: tuple,
                            torch_dev: Optional[torch.device] = torch.device('cpu'),
                            use_toeplitz: Optional[bool] = True) -> dict:
        """
        Computes basis functions for fourier deviations from 
        a regular grid.
        
        Parameters:
        -----------
        mps : torch.Tensor <complex64>
            multi-coil sensitivity maps with shape (C, *im_size)
        matrix_size: tuple
            size of the matrix to compute the eigen-decomp on
        torch_dev : torch.device
            device to compute on
        use_toeplitz : bool
            use toeplitz for faster computation
        
        Returns:
        --------
        dict {
            'temporal_funcs': torch.Tensor <complex64>,
                - has shape (L, *matrix_size)
            'spatial_funcs': torch.Tensor <complex64>,
                - has shape (L, *matrix_size)
            'ks': torch.Tensor <float32>
                - has shape matrix_size with valuse between -0.5 and 0.5
            'rs': torch.Tensor <float32>
                - has shape matrix_size with valuse between -0.5 and 0.5
        }
        basis_funcs : torch.Tensor <complex64>
            temporal basis functions with shape (L, *matrix_size)
        """
        # Resize maps to match matrix size
        mps_rs = spatial_resize(mps, matrix_size, self.spatial_interp_kind).conj()
        C = mps_rs.shape[0]
        
        # Define grids
        ks = gen_grd(matrix_size, balanced=True).to(torch_dev) / self.os_grid
        rs = gen_grd(matrix_size, balanced=True).to(torch_dev)

        # Use other NUFFT to compute SVD quickly
        sp_nufft = sigpy_nufft(matrix_size, oversamp=2.0)
        ks = -sp_nufft.rescale_trajectory(ks)
        nvox = torch.prod(torch.tensor(matrix_size)).item() * (self.os_grid ** len(matrix_size))

        # Define forward, adjoint, graham operators
        def adjoint(x):
            x_coil = sp_nufft.adjoint(x[None,], ks[None,])[0] * nvox ** 0.5
            return (x_coil * mps_rs.conj()).sum(dim=0)
        def forward(y):
            y_coil = y * mps_rs
            return sp_nufft(y_coil[None,], ks[None,])[0] * nvox ** 0.5
        if use_toeplitz:
            kerns = sp_nufft.calc_teoplitz_kernels(ks[None])
            def gram(y):
                y_hat = torch.zeros(matrix_size, device=torch_dev, dtype=complex_dtype)
                batch_size = C
                for c1, c2 in batch_iterator(C, batch_size):
                    y_coil = y * mps_rs[c1:c2]
                    y_coil = sp_nufft.normal_toeplitz(y_coil[None,], kerns)[0] * nvox
                    y_hat += (y_coil * mps_rs[c1:c2].conj()).sum(dim=0)
                return y_hat
        else:
            def gram(y):
                return adjoint(forward(y))
            
        # Compute SVD
        U, S, Vh = svd_power_method_tall(A=forward,
                                         AHA=gram,
                                         niter=100,
                                         inp_dims=matrix_size,
                                         rank=self.L,
                                         device=torch_dev)
        temporal_funcs = rearrange(U * S, 'C ... L -> L C ...').conj()
        spatial_funcs = Vh.conj() # L ...

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
                           grid_new: torch.Tensor,
                           spatial_interp_kind: Optional[str] = 'bicubic') -> torch.Tensor:
        """
        Interpolates basis funcitons.
        
        Parameters:
        -----------
        basis_funcs : torch.Tensor <complex64>
            basis functions with shape (*channels, *matrix_size)
        grid : torch.Tensor <float32>
            spatial grid with shape (*matrix_size, d)
        grid_new : torch.Tensor <float32>
            output spatial grid with shape (..., d)
        spatial_interp_kind : str
            interpolation kind for spatial interpolation
            
        Note:
        -----
        len(matrix_size) must be either 2 or 3, otherwise torch grid_sample doesn't work.
        
        Returns:
        --------
        basis_funcs_new : torch.Tensor <complex64>
            interpolated basis functions with shape (*channels, ...)
        """
        # Consts
        matrix_size = grid.shape[:-1]
        d = len(matrix_size)
        channel_size = basis_funcs.shape[:-d]
        basis_funcs_flt = basis_funcs.flatten(0, -(d+1))
        L = basis_funcs.shape[0]
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
        mode = spatial_interp_kind
        basis_funcs_flt_r = nn.functional.grid_sample(basis_funcs_flt[None,].real, 
                                                      grid=grid_new_flt[None,],
                                                      mode=mode, 
                                                      align_corners=align_corners)[0]
        basis_funcs_flt_i = nn.functional.grid_sample(basis_funcs_flt[None,].imag, 
                                                      grid=grid_new_flt[None,],
                                                      mode=mode, 
                                                      align_corners=align_corners)[0]
        basis_funcs_flt = (basis_funcs_flt_r + 1j * basis_funcs_flt_i).type(complex_dtype)
        
        # Reshape
        basis_funcs_new = basis_funcs_flt.squeeze().reshape((*channel_size, *grid_new.shape[:-1]))
        
        return basis_funcs_new
         
    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        # Grid points to evaluate
        self.ks_eval = trj - (trj*self.os_grid).round()/self.os_grid
        self.rs_eval = gen_grd(self.im_size, balanced=True).to(trj.device)
        
        # Interpolate basis functions
        if self.store_svd_interps:
            dct = self.basis_dict
            self.temporal_funcs = self.interp_basis_funcs(dct['temporal_funcs'], dct['ks'], self.ks_eval, self.spatial_interp_kind)
            self.spatial_funcs = self.interp_basis_funcs(dct['spatial_funcs'], dct['rs'], self.rs_eval, self.spatial_interp_kind)
        
        trj_rs = self.gnufft.rescale_trajectory(trj)
        return trj_rs

    def forward(self, 
                img: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        img : torch.Tensor <complex64>
            input image with shape (N, *img_batch, *im_size)
        trj : torch.Tensor <float32>
            input trajectory with shape (N, *trj_batch, len(im_size))
            
        Returns:
        --------
        ksp : torch.Tensor <complex64>
            output k-space with shape (N, *img_batch, C, *trj_batch)
        """
        
        # To torch
        img_torch, trj_torch = np_to_torch(img, trj)

        # Consts
        d = trj.shape[-1]
        N = trj.shape[0]
        L = self.L
        
        if self.store_svd_interps:
            temporal_funcs = self.temporal_funcs
            spatial_funcs = self.spatial_funcs
        else:
            dct = self.basis_dict
            temporal_funcs = self.interp_basis_funcs(dct['temporal_funcs'], dct['ks'], self.ks_eval, self.spatial_interp_kind)
            spatial_funcs = self.interp_basis_funcs(dct['spatial_funcs'], dct['rs'], self.rs_eval, self.spatial_interp_kind)
        C = temporal_funcs.shape[1]
            

        # Multiply by spatial functions
        empty_dims = img_torch.ndim - spatial_funcs.ndim
        tup = (None, slice(0, L)) + (None,) * empty_dims + (slice(None),) * (spatial_funcs.ndim - 1)
        img_torch = img_torch[:, None, ...] * spatial_funcs[tup]
        # img_torch has shape (N, L, *img_batch, *im_size)
        
        # Gridded part
        ksp_grd = self.gnufft.forward(img_torch, trj_torch)
        # ksp_grd has shape (N, L, *img_batch, *trj_batch)
        #                      (L, C, *trj_batch )

        empty_dims = ksp_grd.ndim - temporal_funcs.ndim 
        tup = (None, slice(0, L), slice(0, C)) + (None,) * empty_dims + (slice(None),) * (temporal_funcs.ndim - 2)
        ksp = (ksp_grd[:, :, None,] * temporal_funcs[tup]).sum(1)
        ksp = ksp.moveaxis(1, 1 + empty_dims)
        
        return ksp * self.os_grid
                    
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
        L = self.L
        os_size = self.padder.pad_im_size

        # Multiply by temporal functions
        empty_dims = ksp.ndim - self.temporal_funcs.ndim
        tup = (None, slice(0, L)) + (None,) * empty_dims + (slice(None),) * (self.temporal_funcs.ndim - 1)
        ksp_torch = ksp_torch[:, None, ...] * self.temporal_funcs.conj()[tup]

        # Adjoint NUFFT
        ksp_os = torch.zeros((*ksp_torch.shape[:-(trj.ndim - 2)], *os_size), 
                             dtype=complex_dtype, device=ksp_torch.device)
        for i in range(N):
            ksp_os[i] = multi_grid(ksp_torch[i], trj_torch[i].type(torch.int32), os_size)
        img_os = ifft(ksp_os, dim=tuple(range(-d, 0)))
        img = self.padder.adjoint(img_os)

        empty_dims = img.ndim - self.spatial_funcs.ndim - 1
        tup = (None, slice(0, L)) + (None,) * empty_dims + (slice(None),) * (self.spatial_funcs.ndim - 1)
        img = (img * self.spatial_funcs.conj()[tup]).sum(1)

        return img * self.os_grid
