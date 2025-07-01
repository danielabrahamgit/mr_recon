import torch
import torch.nn as nn
import numpy as np
import sigpy as sp

from dataclasses import dataclass
from mr_recon import dtypes
from mr_recon.fourier import fft, ifft
from mr_recon.utils import batch_iterator, np_to_torch, torch_to_np
from mr_recon.pad import PadLast
from mr_recon.spatial import resize
from mr_recon.fourier import (
    gridded_nufft,
    sigpy_nufft,
    torchkb_nufft,
    NUFFT
)
from einops import rearrange, einsum
from typing import Optional
from tqdm import tqdm

@dataclass
class batching_params:
    coil_batch_size: Optional[int] = 1
    sub_batch_size: Optional[int] = 1
    field_batch_size: Optional[int] = 1
    toeplitz_batch_size: Optional[int] = 1
    img_batch_size: Optional[int] = 1

class linop(nn.Module):
    """
    Generic linop
    """

    def __init__(self, ishape, oshape):
        super().__init__()
        self.ishape = ishape
        self.oshape = oshape
    
    def forward(self, *args):
        raise NotImplementedError
    
    def adjoint(self, *args):
        raise NotImplementedError
    
    def normal(self, *args):
        raise NotImplementedError

class type3_nufft_naive(linop):
    """
    Impliments the following linop using naive summning:
    y(t) = sum_r x(r) e^{-j 2\pi phi(r) \cdot alpha(t)}
    """
    
    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor):
        """
        Args:
        -----
        phis : torch.Tensor
            The spatial phase maps, shape (B, *im_size)
        alphas : torch.Tensor
            The temporal phase coefficients, shape (B, *trj_size)
        """
        super().__init__(phis.shape[1:], alphas.shape[1:])
        B = phis.shape[0]
        assert phis.shape[0] == alphas.shape[0], "phis and alphas must have same number of bases (B)"
        self.phis = phis.reshape((B, -1))
        self.alphas = alphas.reshape((B, -1))
        
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
        x : torch.Tensor
            The image to be transformed, shape (N, *im_size)
        
        Returns:
        --------
        y : torch.Tensor
            The output with shape (N, *trj_size)
        """
        N = x.shape[0]
        enc_mat = torch.exp(-2j * torch.pi * (self.phis.T @ self.alphas))
        x_flt = x.reshape((N, -1))
        return (x_flt @ enc_mat).reshape((N, *self.oshape))
        
    def adjoint(self, 
                y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
        y : torch.Tensor
            The output with shape (N, *trj_size)
        
        Returns:
        --------
        x : torch.Tensor
            The image with shape (N, *im_size)
        """
        N = y.shape[0]
        enc_mat = torch.exp(2j * torch.pi * (self.alphas.T @ self.phis))
        y_flt = y.reshape((N, -1))
        return (y_flt @ enc_mat).reshape((N, *self.ishape))
    
    def normal(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        -----
        x : torch.Tensor
            The image to be transformed, shape (N, *im_size)
        
        Returns:
        --------
        y : torch.Tensor
            The output with shape (N, *trj_size)
        """
        return self.adjoint(self.forward(x))

class type3_nufft(linop):
    
    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 oversamp: float = 2.0,
                 width: float = 4.0,
                 use_toep: Optional[bool] = False):
        """
        KB implimentation of the type-3 nufft as described in:
        "A PARALLEL NONUNIFORM FAST FOURIER TRANSFORM LIBRARY 
        BASED ON AN ``EXPONENTIAL OF SEMICIRCLE" KERNEL - Barnett et. al.
        "https://epubs.siam.org/doi/pdf/10.1137/18M120885X
        
        type3 nuffts look like:
            y_k = sum_n x_n e^{-j 2\pi phi_n * alpha_k}
            
        TODO list: 
        ----------
        1. Fix KB beta calculation -- beatty is not good for large oversamp factors
        2. Update gridding/interp functions to be n-dimensional (currently up to 5d)

        Args:
        -----
        phis : torch.Tensor
            The spatial phase maps, shape (B, *im_size)
        alphas : torch.Tensor
            The temporal phase coefficients, shape (B, *trj_size)
        oversamp : float
            The oversampling factor for spatial gridding
        width : float
            The width of the gridding kernel
        use_toep : bool
            toggles Topelitz for gram/normal/AHA operator
        """    
        # Consts
        B, *im_size = phis.shape
        B_, *trj_size = alphas.shape
        R = np.prod(im_size)
        T = np.prod(trj_size)
        torch_dev = phis.device
        assert B == B_, "phis and alphas must have same number of bases (B)"
        assert phis.device == alphas.device, "phis and alphas must be on same device"
        super().__init__(tuple(im_size), tuple(trj_size))
        
        # Flatten everything
        phis_flt = phis.reshape((B, R))
        alphas_flt = alphas.reshape((B, T))
        
        # Center alphas and phis
        phis_mp = (phis_flt.min(dim=1).values + phis_flt.max(dim=1).values)/2
        alphas_mp = (alphas_flt.min(dim=1).values + alphas_flt.max(dim=1).values)/2
        phis_flt_cent = phis_flt - phis_mp[:, None]
        alphas_flt_cent = alphas_flt - alphas_mp[:, None]
        
        # Rescale phis to be between [-1/2, 1/2]
        scales = phis_flt_cent.abs().max(dim=1).values * 2
        phis_flt_cent /= scales[:, None]
        phis_mp /= scales
        alphas_flt_cent *= scales[:, None]
        alphas_mp *= scales
        
        # Store phis, alphas, other constants
        self.phis = phis_flt_cent
        self.alphas = alphas_flt_cent
        self.phis_mp = phis_mp
        self.alphas_mp = alphas_mp
        self.torch_dev = torch_dev
        self.B = B
        
        # Consts for gridding
        self.grd_S = self.alphas.abs().max(dim=1).values
        self.grd_W = width
        self.grd_N_os = torch.ceil(2 * self.grd_S * oversamp + self.grd_W).long()
        self.grd_os = self.grd_N_os / (self.grd_N_os / oversamp).round() # FIXME?
        self.grd_N = self.grd_N_os / self.grd_os
        self.grd_gamma = self.grd_N_os / (2 * self.grd_os * self.grd_S)
        self.grd_beta = np.pi * (((self.grd_W / self.grd_os) * (self.grd_os - 0.5))**2 - 0.8)**0.5
        self.grd_N_os = tuple((self.grd_N * self.grd_os).ceil().long().tolist())
        self.nft = sigpy_nufft(self.grd_N_os)
        # print(f'Actual oversamplings = {self.grd_os}')
        # print(f'Oversampling Grids = {self.grd_N_os}')
        
        if use_toep:
            # print('Computing toeplitz Kernels ... ', end='')
            apod_weights = self.apod(self.alphas.T * self.grd_gamma / self.grd_N / self.grd_os, self.grd_beta, self.grd_W).prod(dim=-1)
            self.kerns = self.nft.calc_teoplitz_kernels(trj=self.alphas.T[None,] * self.grd_gamma, weights=apod_weights[None,] ** 2)
            # print('done.')
        else:
            self.kerns = None
        
    @staticmethod
    def apod(x, beta, W):
        a = (beta**2 - (np.pi * W * x) ** 2) ** 0.5
        return a / torch.sinh(a)
        
    def forward(self,
                x: torch.Tensor,) -> torch.Tensor:
        """
        Forward type 3 nufft.
        
        Args:
        -----
        x : torch.Tensor
            The image to be transformed, shape (N, *im_size)
        
        Returns:
        --------
        y : torch.Tensor
            The output with shape (N, *trj_size)
        """
        # Consts
        N = x.shape[0]
        
        # ----------------- Step 0: Apply spatial midpoints -----------------
        phz = self.phis.T @ self.alphas_mp
        x_mp = x.reshape(N, -1) * torch.exp(-2j * torch.pi * phz)

        # ----------------- Step 1: gridding in the image domain -----------------
        # Compute shifts and scales
        scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
        shifts = (self.grd_os * self.grd_N).ceil() // 2
        
        # Define output matrix size and betas
        betas = tuple(self.grd_beta.tolist())

        # Move to cupy
        x_cp_flt = torch_to_np(x_mp)
        dev = sp.get_device(x_cp_flt)
        with dev:
            
            # Rescale trajectory
            trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
            trj_cp = torch_to_np(trj)
            
            # Gridding
            output = sp.interp.gridding(x_cp_flt, trj_cp, (N,) + self.grd_N_os,
                                        kernel='kaiser_bessel', width=self.grd_W, param=betas)
            x_grid = np_to_torch(output)
            x_grid /= self.grd_W ** self.B
            
        # ----------------- Step 2: Call NUFFT on gridded data -----------------
        alphas_rep = torch.repeat_interleave(self.alphas.T[None,], N, dim=0) # N T B
        y_pre_apod = self.nft.forward(x_grid, alphas_rep * self.grd_gamma) # N T
        y_pre_apod *= np.prod(self.grd_N_os) ** 0.5
        
        # ----------------- Step 3: Apodize -----------------
        y = y_pre_apod * self.apod(alphas_rep * self.grd_gamma / self.grd_N / self.grd_os, self.grd_beta, self.grd_W).prod(dim=-1)
        
        # ----------------- Step 4: Apply temporal midpoints -----------------
        phz = (self.alphas.T + self.alphas_mp) @ (self.phis_mp)
        y = y * torch.exp(-2j * torch.pi * phz)
        
        # ----------------- Step 5: Reshape and Pray -----------------
        return y.reshape((N, *self.oshape))

    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint type 3 nufft.
        
        Args:
        -----
        y : torch.Tensor
            The k-space data to be transformed, shape (N, *trj_size)
        
        Returns:
        --------
        x : torch.Tensor
            The output with shape (N, *im_size)
        """
        # Consts
        N = y.shape[0]
        
        # ----------------- Step 0: Apply temporal midpoints -----------------
        phz = self.alphas.T @ self.phis_mp
        y_mp = y.reshape((N, -1)) * torch.exp(2j * torch.pi * phz)
        
        # ----------------- Step 1: Apodize -----------------
        alphas_rep = torch.repeat_interleave(self.alphas.T[None,], N, dim=0) # N T B
        y_apod = y_mp * self.apod(alphas_rep * self.grd_gamma / self.grd_N / self.grd_os, self.grd_beta, self.grd_W).prod(dim=-1)
        
        # ----------------- Step 2: Call Adjoint NUFFT -----------------
        x_grid = self.nft.adjoint(y_apod, alphas_rep * self.grd_gamma) # N *self.grd_N_os
        x_grid *= np.prod(self.grd_N_os) ** 0.5
        
        # ----------------- Step 3: Interpolation in the image domain -----------------
        # Compute shifts and scales
        scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
        shifts = (self.grd_os * self.grd_N).ceil() // 2
        
        # Define output matrix size and betas
        betas = tuple(self.grd_beta.tolist())

        # Move to cupy
        x_grid_cp = torch_to_np(x_grid)
        dev = sp.get_device(x_grid_cp)
        with dev:
            
            # Rescale trajectory
            trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
            trj_cp = torch_to_np(trj)
            
            # Interpolate
            output = sp.interp.interpolate(x_grid_cp, trj_cp,
                                           kernel='kaiser_bessel', width=self.grd_W, param=betas)
            x = np_to_torch(output)
            x /= self.grd_W ** self.B

        # ----------------- Step 4: Apply spatial midpoints -----------------
        phz = (self.phis.T + self.phis_mp) @ self.alphas_mp
        x = x * torch.exp(2j * torch.pi * phz)
        
        # ----------------- Step 5: Reshape and pray -----------------
        return x.reshape((N, *self.ishape))

    def normal(self,
               x: torch.Tensor) -> torch.Tensor:
        """
        Applies normal operator
        
        Args:
        -----
        x : torch.Tensor
            The image to be transformed, shape (N, *im_size)
            
        Returns:
        --------
        torch.Tensor
            The output with shape (N, *im_size)
        """
        if self.kerns is None:
            return self.adjoint(self.forward(x))
        else:
            # Consts
            N = x.shape[0]
            B = self.phis.shape[0]
            
            # ----------------- Step 0: Apply spatial midpoints -----------------
            phz = (self.phis.T + self.phis_mp) @ self.alphas_mp
            x_mp = x.reshape(N, -1) * torch.exp(-2j * torch.pi * phz)

            # ----------------- Step 1: gridding in the image domain -----------------
            # Compute shifts and scales
            scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
            shifts = (self.grd_os * self.grd_N).ceil() // 2
            
            # Define output matrix size and betas
            betas = tuple(self.grd_beta.tolist())

            # Move to cupy
            x_cp_flt = torch_to_np(x_mp)
            dev = sp.get_device(x_cp_flt)
            with dev:
                
                # Rescale trajectory
                trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
                trj_cp = torch_to_np(trj)
                
                # Gridding
                output = sp.interp.gridding(x_cp_flt, trj_cp, (N,) + self.grd_N_os,
                                                kernel='kaiser_bessel', width=self.grd_W, param=betas)
                x_grid = np_to_torch(output)
                x_grid /= self.grd_W ** self.B
                
            # ----------------- Step 2: Apply toeplitz kernels -----------------
            x_zp = resize(x_grid, [N,] + [self.grd_N_os[i] * 2 for i in range(B)])
            x_ft = fft(x_zp, dim=tuple(range(-B, 0)))
            x_tp = x_ft * self.kerns
            x_ift = ifft(x_tp, dim=tuple(range(-B, 0)))
            x_crp = resize(x_ift, (N, *self.grd_N_os))
            x_grid = x_crp
            x_grid *= np.prod(self.grd_N_os)
            
            # ----------------- Step 3: Interpolation in the image domain -----------------
            # Compute shifts and scales
            scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
            shifts = (self.grd_os * self.grd_N).ceil() // 2
            
            # Define output matrix size and betas
            betas = tuple(self.grd_beta.tolist())

            # Move to cupy
            x_grid_cp = torch_to_np(x_grid)
            dev = sp.get_device(x_grid_cp)
            with dev:
                
                # Rescale trajectory
                trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
                trj_cp = torch_to_np(trj)
                
                # Interpolate
                output = sp.interp.interpolate(x_grid_cp, trj_cp,
                                               kernel='kaiser_bessel', width=self.grd_W, param=betas)
                x = np_to_torch(output)
                x /= self.grd_W ** self.B

            # ----------------- Step 4: Apply spatial midpoints -----------------
            phz = (self.phis.T + self.phis_mp) @ self.alphas_mp
            x = x * torch.exp(2j * torch.pi * phz)
            
            # ----------------- Step 5: Reshape and pray -----------------
            return x.reshape((N, *self.ishape))

class imperf_coil_lowrank(linop):
    """
    Linop for combining imperfections and coils into
    a single low rank operator
    """
    
    def __init__(self,
                 trj: torch.Tensor,
                 spatial_funcs: torch.Tensor,
                 temporal_funcs: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        trj : torch.tensor
            The k-space trajectory with shape (*trj_size, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        spatial_funcs : torch.tensor
            the spatial basis functions for imperfection models with shape (L, *im_size)
        temporal_funcs : torch.tensor
            the temporal basis functions for imperfection models with shape (L, C, *trj_size)
        dcf : torch.tensor
            the density comp. functon with shape (*trj_size)
        nufft : NUFFT
            the nufft object, defaults to sigpy_nufft
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains various batch sizes
        """
        im_size = spatial_funcs.shape[1:]
        oshape = temporal_funcs.shape[1:]
        super().__init__(im_size, oshape)

        # Consts
        L = spatial_funcs.shape[0]
        C = temporal_funcs.shape[1]
        torch_dev = trj.device
        assert L == temporal_funcs.shape[0]
        assert temporal_funcs.device == torch_dev
        assert spatial_funcs.device == torch_dev

        # Default params
        if nufft is None:
            nufft = sigpy_nufft(im_size)
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=dtypes.real_dtype, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Rescale and change types
        trj = nufft.rescale_trajectory(trj).type(dtypes.real_dtype)
        dcf = dcf.type(dtypes.real_dtype)
        spatial_funcs = spatial_funcs.type(dtypes.complex_dtype)
        temporal_funcs = temporal_funcs.type(dtypes.complex_dtype)
        
        # Compute toeplitz kernels
        if use_toeplitz:
            weights = einsum(temporal_funcs.conj(), temporal_funcs, 'L1 C ..., L2 C ... -> L1 L2 ...')
            weights = rearrange(weights, 'L1 L2 ... -> (L1 L2) ... ') * dcf[None, ...]
            self.toep_kerns = None
            for b1 in range(0, weights.shape[0], bparams.toeplitz_batch_size):
                b2 = min(b1 + bparams.toeplitz_batch_size, weights.shape[0])
                toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], weights[b1:b2])
                if self.toep_kerns is None:
                    self.toep_kerns = toep_kerns
                else:
                    self.toep_kerns = torch.cat((self.toep_kerns, toep_kerns), dim=0)
            self.toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], weights) 
            self.toep_kerns = rearrange(self.toep_kerns, '(n1 n2) ... -> n1 n2 ...', n1=L, n2=L)
        else:
            self.toep_kerns = None

        # Save
        self.L = L
        self.C = C
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.spatial_funcs = spatial_funcs
        self.temporal_funcs = temporal_funcs
        self.trj = trj
        self.dcf = dcf
        self.nufft = nufft
        self.bparams = bparams
        self.torch_dev = torch_dev

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        img : torch.tensor
            the image with shape (*im_size)
        
        Returns
        ---------
        ksp : torch.tensor
            the k-space data with shape (C, *trj_size)
        """

        # Useful constants
        C = self.C
        L = self.L
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((C, *self.trj.shape[:-1]), dtype=dtypes.complex_dtype, device=self.torch_dev)

        # Batch over segments
        for l1, l2 in batch_iterator(L, seg_batch_size):
            Bx = img * self.spatial_funcs[l1:l2]

            # NUFFT
            FBx = self.nufft.forward(Bx[None,], self.trj[None,])[0] # L *trj_size

            # Batch over coils
            for c1, c2 in batch_iterator(C, coil_batch_size):

                # Temporal functions
                HFBx = einsum(FBx, self.temporal_funcs[l1:l2, c1:c2,], 'L ..., L C ... -> C ...')
                
                # Append to k-space
                ksp[c1:c2, ...] += HFBx

        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Adjoint call of this linear model.

        Parameters
        ----------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        
        Returns
        ---------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        """

        # Useful constants
        C = self.C
        L = self.L
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result image
        img = torch.zeros(self.im_size, dtype=dtypes.complex_dtype, device=self.torch_dev)

        # Batch over coils
        for c1, c2 in batch_iterator(C, coil_batch_size):

            # DCF
            Dy = ksp[c1:c2, ...] * self.dcf[None, ...]
            
            # Batch over segments
            for l1, l2 in batch_iterator(L, seg_batch_size):

                # Adjoint temporal functions
                HDy = einsum(Dy, self.temporal_funcs[l1:l2, c1:c2,].conj(), 'C ... , L C ... -> L ...')

                # Adjoint nufft
                FHDy = self.nufft.adjoint(HDy[None,], self.trj[None,])[0] # L *im_size

                # Adjoint spatial functions
                BHFDy = einsum(FHDy, self.spatial_funcs[l1:l2].conj(), 'L ... , L ... -> ...')

                # Append to image
                img += BHFDy

        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Gram/normal call of this linear model (A.H (A (x))).

        Parameters
        ----------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        img_hat : torch.tensor <complex> | GPU
            the ouput image with shape (*im_size)
        """
        
        # Do forward and adjoint, a bit slow
        if not self.use_toeplitz:
            return self.adjoint(self.forward(img))
        else:

            # Useful constants
            L = self.L
            dim = len(self.im_size)
            seg_batch_size = self.bparams.field_batch_size

            # Padding operator
            im_size_os = self.toep_kerns.shape[-dim:]
            padder = PadLast(im_size_os, self.im_size)

            # Result array
            img_hat = torch.zeros_like(img)
                    
            # Batch over segments
            for l1, l2 in batch_iterator(L, seg_batch_size):

                # Apply spatial functions
                Bx = img * self.spatial_funcs[l1:l2]

                # Oversampled FFT
                Bx = padder.forward(Bx)
                FBx = fft(Bx, dim=tuple(range(-dim, 0))) # L *im_size_os

                # Apply Toeplitz kernels
                MFBx = einsum(self.toep_kerns[:, l1:l2, ...], FBx,
                              'Lo Li ..., Li ... -> Lo ...')
                
                # Inverse FFT
                FMFBx = ifft(MFBx, dim=tuple(range(-dim, 0))) # L *im_size_os
                FMFBx = padder.adjoint(FMFBx)

                # Update output
                img_hat += FMFBx
        
        return img_hat

class multi_chan_linop(linop):
    """
    Linop for doing channel by channel reconstruction
    """

    def __init__(self,
                 out_size: tuple,
                 trj: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 use_toeplitz: Optional[bool] = False,
                 spatial_funcs: Optional[torch.Tensor] = None,
                 temporal_funcs: Optional[torch.Tensor] = None,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        out_size : tuple 
            coil and image dims as tuple of ints (nc, dim1, dim2, ...)
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (*trj_size, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        mps : torch.tensor <complex> | GPU
            sensititvity maps with shape (ncoil, *im_size)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (*trj_size)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        use_toeplitz : bool
            toggles toeplitz normal operator
        spatial_funcs : torch.tensor <complex> | GPU
            the spatial basis functions for imperfection models with shape (L, *im_size)
        temporal_funcs : torch.tensor <complex> | GPU
            the temporal basis functions for imperfection models with shape (L, *trj_size)
        bparams : batching_params
            contains various batch sizes
        """
        ishape = out_size
        oshape = (out_size[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)

        mps_dummy = torch.ones((1, *out_size[1:]), dtype=dtypes.complex_dtype, device=trj.device)
        self.A = sense_linop(trj, mps_dummy, dcf, nufft, 
                             spatial_funcs=spatial_funcs,
                             temporal_funcs=temporal_funcs,
                             use_toeplitz=use_toeplitz, bparams=bparams)

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        img : torch.tensor <complex> | GPU
            the multi-channel image with shape (nc, *im_size)
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        """
        ksp = torch.zeros(self.oshape, dtype=dtypes.complex_dtype, device=img.device)
        for i in range(self.ishape[0]):
            ksp[i] = self.A.forward(img[i])[0]
        return ksp

    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Adjoint call of this linear model.

        Parameters
        ----------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        
        Returns
        ---------
        img : torch.tensor <complex> | GPU
            the multi channel image with shape (nc, *im_size)
        """
        img = torch.zeros(self.ishape, dtype=dtypes.complex_dtype, device=ksp.device)
        for i in range(self.ishape[0]):
            img[i] = self.A.adjoint(ksp[i][None,])
        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Gram/normal call of this linear model (A.H (A (x))).

        Parameters
        ----------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        img_hat : torch.tensor <complex> | GPU
            the ouput image with shape (*im_size)
        """
        img_hat = torch.zeros(self.ishape, dtype=dtypes.complex_dtype, device=img.device)
        for i in range(self.ishape[0]):
            img_hat[i] = self.A.normal(img[i])
        return img_hat

class coil_spatiotemporal_linp(linop):
    
    def __init__(self,
                 trj: torch.Tensor,
                 spatial_funcs: torch.Tensor,
                 temporal_coil_funcs: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (*trj_size, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        spatial_funcs : torch.tensor <complex> | GPU
            the spatial functions for imperfection models with shape (L, *im_size)
        temporal_coil_funcs : torch.tensor <complex> | GPU
            the temporal functions for imperfection models with shape (L, C, *trj_size)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (*trj_size)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains various batch sizes
        """
        im_size = spatial_funcs.shape[1:]
        trj_size = trj.shape[:-1]
        C = temporal_coil_funcs.shape[1]
        super().__init__(im_size, (C, *trj_size))

        # Consts
        torch_dev = trj.device
        assert spatial_funcs.device == torch_dev
        assert temporal_coil_funcs.device == torch_dev
        assert temporal_coil_funcs.shape[0] == spatial_funcs.shape[0]

        # Default params
        if nufft is None:
            nufft = sigpy_nufft(im_size)
        if dcf is None:
            dcf = torch.ones(trj_size, dtype=dtypes.real_dtype, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Rescale and change types
        trj = nufft.rescale_trajectory(trj).type(dtypes.real_dtype)
        dcf = dcf.type(dtypes.real_dtype)
        
        # Compute toeplitz kernels
        if use_toeplitz:
            raise NotImplementedError("Toeplitz kernels not implemented for coil imperfection models")

        # Save
        self.b = spatial_funcs
        self.h = temporal_coil_funcs
        self.im_size = im_size
        self.trj_size = trj_size
        self.trj = trj
        self.dcf = dcf
        self.nufft = nufft
        self.bparams = bparams
        self.torch_dev = torch_dev

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (C, *trj_size)
        """

        # Useful constants
        L = self.b.shape[0]
        C = self.h.shape[1]
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((C, *self.trj_size), dtype=dtypes.complex_dtype, device=self.torch_dev)

        # Batch over segments 
        for l1, l2 in batch_iterator(L, seg_batch_size):    
            
            # Apply spatial functions
            Bx = self.b[l1:l2] * img # L *im_size

            # NUFFT
            FBx = self.nufft.forward(Bx[None,], self.trj[None, ...])[0] # L *trj_size
            
            # Batch over coils ??
            # Apply temporal functions
            HFBx = (FBx[:, None] * self.h[l1:l2]).sum(dim=0) # C *trj_size
            
            # Append to k-space
            ksp += HFBx

        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Adjoint call of this linear model.

        Parameters
        ----------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        
        Returns
        ---------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        """

        # Useful constants
        L = self.b.shape[0]
        C = self.h.shape[1]
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result image
        img = torch.zeros(self.im_size, dtype=dtypes.complex_dtype, device=self.torch_dev)
        
        # Apply DCF
        Wy = (ksp * self.dcf) # C *trj_size
            
        # Batch over segments
        for l1, l2 in batch_iterator(L, seg_batch_size):
            
            # batch over coils?    
            # Apply temporal functions
            HWy = (Wy * self.h.conj()[l1:l2, :]).sum(dim=1) # L *trj_size
            
            # Adjoint NUFFT
            FHWy = self.nufft.adjoint(HWy[None, ...], self.trj[None, ...])[0] # L *im_size

            # Adjoint spatial functions
            BFHWy = (FHWy * self.b.conj()[l1:l2]).sum(dim=0) # im_size

            # Append to image
            img += BFHWy

        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Gram/normal call of this linear model (A.H (A (x))).

        Parameters
        ----------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        img_hat : torch.tensor <complex> | GPU
            the ouput image with shape (*im_size)
        """
        return self.adjoint(self.forward(img))

class sense_linop(linop):
    """
    Linop for sense models with optional imperfection modeling
    """
    
    def __init__(self,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 use_toeplitz: Optional[bool] = False,
                 spatial_funcs: Optional[torch.Tensor] = None,
                 temporal_funcs: Optional[torch.Tensor] = None,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (*trj_size, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        mps : torch.tensor <complex> | GPU
            sensititvity maps with shape (ncoil, *im_size)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (*trj_size)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains various batch sizes
        spatial_funcs : torch.tensor <complex> | GPU
            the spatial functions for imperfection models with shape (L, *im_size)
        temporal_funcs : torch.tensor <complex> | GPU
            the temporal functions for imperfection models with shape (L, *trj_size)
        """
        im_size = mps.shape[1:]
        trj_size = trj.shape[:-1]
        ncoils = mps.shape[0]
        super().__init__(im_size, (ncoils, *trj_size))

        # Consts
        torch_dev = trj.device
        assert mps.device == torch_dev

        # Default params
        if nufft is None:
            nufft = sigpy_nufft(im_size)
        if dcf is None:
            dcf = torch.ones(trj_size, dtype=dtypes.real_dtype, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        if spatial_funcs is None:
            spatial_funcs = torch.ones((1,)*(len(im_size)+1), dtype=dtypes.complex_dtype, device=torch_dev)
        if temporal_funcs is None:
            temporal_funcs = torch.ones((1,)*(len(trj_size)+1), dtype=dtypes.complex_dtype, device=torch_dev)
        
        # Rescale and change types
        trj = nufft.rescale_trajectory(trj).type(dtypes.real_dtype)
        dcf = dcf.type(dtypes.real_dtype)
        mps = mps.type(dtypes.complex_dtype)
        
        # Compute toeplitz kernels
        if use_toeplitz:
            if (spatial_funcs is None) or (temporal_funcs is None):
                self.toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], dcf[None,])[0]
            else:
                weights = einsum(temporal_funcs.conj(), temporal_funcs, 'L1 ... , L2 ... -> L1 L2 ...')
                weights = rearrange(weights, 'n1 n2 ... -> (n1 n2) ... ') * dcf[None, ...]
                self.toep_kerns = None
                for b1 in range(0, weights.shape[0], bparams.toeplitz_batch_size):
                    b2 = min(b1 + bparams.toeplitz_batch_size, weights.shape[0])
                    toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], weights[b1:b2])
                    if self.toep_kerns is None:
                        self.toep_kerns = toep_kerns
                    else:
                        self.toep_kerns = torch.cat((self.toep_kerns, toep_kerns), dim=0)
                self.toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], weights) 
                L = temporal_funcs.shape[0]
                self.toep_kerns = rearrange(self.toep_kerns, '(n1 n2) ... -> n1 n2 ...', n1=L, n2=L)
        else:
            self.toep_kerns = None

        # Save
        self.im_size = im_size
        self.trj_size = trj_size
        self.use_toeplitz = use_toeplitz
        self.trj = trj
        self.mps = mps
        self.dcf = dcf
        self.nufft = nufft
        self.bparams = bparams
        self.spatial_funcs = spatial_funcs
        self.temporal_funcs = temporal_funcs
        self.torch_dev = torch_dev

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (C, *trj_size)
        """

        # Useful constants
        imperf_rank = self.spatial_funcs.shape[0]
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((nc, *self.trj_size), dtype=dtypes.complex_dtype, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps_times_img = self.mps[c:d] * img

            # Batch over segments 
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):

                # Apply Spatial functions
                SBx = mps_times_img[:, None, ...] * self.spatial_funcs[l1:l2]

                # NUFFT and temporal terms
                FSBx = self.nufft.forward(SBx[None,], self.trj[None, ...])[0]
                
                # Apply temporal functions
                HFSBx = (FSBx * self.temporal_funcs[l1:l2]).sum(dim=-len(self.trj_size)-1)
                
                # Append to k-space
                ksp[c:d, ...] += HFSBx

        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Adjoint call of this linear model.

        Parameters
        ----------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        
        Returns
        ---------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        """

        # Useful constants
        imperf_rank = self.temporal_funcs.shape[0]
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result image
        img = torch.zeros(self.im_size, dtype=dtypes.complex_dtype, device=self.torch_dev)
            
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            ksp_weighted = ksp[c:d, ...] * self.dcf[None, ...]

            # Batch over segments
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                
                # Adjoint temporal functions
                HWy = ksp_weighted[:, None, ...] * self.temporal_funcs[l1:l2].conj()
                
                # Adjoint NUFFT
                FHWy = self.nufft.adjoint(HWy[None, ...], self.trj[None, ...])[0] # C L *im_size

                # Adjoint coil maps
                SFHWy = einsum(FHWy, mps.conj(), 'nc nseg ..., nc ... -> nseg ...')

                # Adjoint spatial functions
                BSFHWy = (SFHWy * self.spatial_funcs[l1:l2].conj()).sum(dim=-len(self.im_size)-1)

                # Append to image
                img += BSFHWy

        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Gram/normal call of this linear model (A.H (A (x))).

        Parameters
        ----------
        img : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        img_hat : torch.tensor <complex> | GPU
            the ouput image with shape (*im_size)
        """
        
        # Do forward and adjoint, a bit slow
        if not self.use_toeplitz:
            return self.adjoint(self.forward(img))
        else:

            # Useful constants
            imperf_rank = self.temporal_funcs.shape[0]
            nc = self.mps.shape[0]
            dim = len(self.im_size)
            coil_batch_size = self.bparams.coil_batch_size
            seg_batch_size = self.bparams.field_batch_size

            # Padding operator
            im_size_os = self.toep_kerns.shape[-dim:]
            padder = PadLast(im_size_os, self.im_size)

            # Result array
            img_hat = torch.zeros_like(img)
                    
            # Batch over coils
            for c, d in batch_iterator(nc, coil_batch_size):
                mps = self.mps[c:d]

                # Apply Coils
                Sx = mps * img

                # Batch over segments
                for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):

                    # Apply spatial funcs
                    SBx = Sx[:, None, ...] * self.spatial_funcs[l1:l2]

                    # Apply zero-padded FFT
                    RSBx = padder.forward(SBx)
                    FSBx = fft(RSBx, dim=tuple(range(-dim, 0))) # nc nseg *im_size_os

                    # Apply Toeplitz kernels
                    MFSBx = einsum(self.toep_kerns[:, l1:l2, ...],  FSBx,
                                   'L L2 ..., C L2 ... -> C L ...')
                    
                    # Apply iFFT and mask
                    FMFBSx = ifft(MFSBx, dim=tuple(range(-dim, 0))) 
                    FMFBSx = padder.adjoint(FMFBSx)

                    # Adjoint spatial funcs
                    BFMFBSx = (FMFBSx * self.spatial_funcs[l1:l2].conj()).sum(dim=-len(self.im_size)-1)

                    # Apply adjoint mps
                    RFMFBSx = (BFMFBSx * mps.conj()).sum(dim=0)
                    
                    # Update output
                    img_hat += RFMFBSx
        
        return img_hat

class sense_linop_noise_cov(linop):
    
    def __init__(self,
                 noise_cov,
                 **sense_args):
        """
        Args:
        -----
        noise_cov : torch.tensor
            the k-space noise covariance matrix with shape (..., nc, nc)
        **sense_args : dict
            the arguments for the sense_linop constructor
        """
        im_size = sense_args['mps'].shape[1:]
        trj_size = sense_args['trj'].shape[:-1]
        C = sense_args['mps'].shape[0]
        super().__init__(im_size, (C, *trj_size))
        
        self.set_noise_cov(noise_cov)
        self.A = sense_linop(**sense_args)
    
    def apply_kspace_coil_mat(self,
                              ksp: torch.Tensor,
                              ksp_coil_mat: torch.Tensor) -> torch.Tensor:
        """
        Applies a coil matrix to each point in kspace
        
        Parameters
        ----------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        ksp_coil_mat : torch.tensor <complex> | GPU
            the coil matrix with shape (*trj_size, nc, nc)
        
        Returns
        ---------
        ksp_new : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size) after applying the coil matrix
        """
        ksp_new = torch.zeros_like(ksp)
        first_coil_batch = self.A.bparams.coil_batch_size
        for c1, c2 in batch_iterator(ksp.shape[0], first_coil_batch):
            second_coil_batch = ksp.shape[0]
            for d1, d2 in batch_iterator(ksp.shape[0], second_coil_batch):
                ksp_new[c1:c2] = einsum(ksp[d1:d2], ksp_coil_mat[..., c1:c2, d1:d2], 'ci ..., ... co ci -> co ...')
        # ksp_new = einsum(ksp, ksp_coil_mat, 'ci ..., ... co ci -> co ...')
        return ksp_new  
    
    def set_noise_cov(self, noise_cov: torch.Tensor):
        if noise_cov is None:
            self.inv_noise_cov = None
        else:
            self.inv_noise_cov = torch.zeros_like(noise_cov)
            for i in range(noise_cov.shape[0]):
                self.inv_noise_cov[i] = torch.linalg.inv(noise_cov[i])
            
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.A.forward(img)
    
    def adjoint(self, ksp: torch.Tensor) -> torch.Tensor:
        if self.inv_noise_cov is not None:
            ksp_new = self.apply_kspace_coil_mat(ksp, self.inv_noise_cov)
        else:
            ksp_new = ksp
        return self.A.adjoint(ksp_new)
    
    def normal(self, img: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(img))

class subspace_linop_new(linop):
    
    def __init__(self,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 use_toeplitz: Optional[bool] = False,
                 spatial_funcs: Optional[torch.Tensor] = None,
                 temporal_funcs: Optional[torch.Tensor] = None,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        trj : torch.tensor 
            The k-space trajectory with shape (*trj_size, d)
            trj_size = (..., T), the T is the number of time points for subspace
        phi : torch.tensor
            subspace basis with shape (K, T)
        mps : torch.tensor
            sensititvity maps with shape (C, *im_size)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (*trj_size)
        nufft : NUFFT
            the nufft object, defaults to sigpy_nufft
        use_toeplitz : bool
            toggles toeplitz normal operator
        spatial_funcs : torch.tenso
            the spatial functions for imperfection models with shape (L, *im_size)
        temporal_funcs : torch.tensor
            the temporal functions for imperfection models with shape (L, *trj_size)
        bparams : batching_params
            contains the batch sizes for the coils, subspace coeffs, and field segments
        """
        im_size = mps.shape[1:]
        ishape = (phi.shape[0], *im_size)
        oshape = (mps.shape[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)
        
        if use_toeplitz is True:
            raise NotImplementedError("Toeplitz normal operator not implemented for subspace models")
        
        assert phi.shape[1] == trj.shape[-2]
        self.A = sense_linop(trj=trj, mps=mps, dcf=dcf, 
                             nufft=nufft,
                             use_toeplitz=False,
                             spatial_funcs=spatial_funcs,
                             temporal_funcs=temporal_funcs,
                             bparams=bparams)
        self.phi = phi
        
    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        alphas : torch.tensor 
            the subspace coefficient volumes with shape (K, *im_size)
        
        Returns
        ---------
        ksp : torch.tensor
            the k-space data with shape (C, *trj_size)
        """
        K = self.phi.shape[0]
        ksp = torch.zeros(self.oshape, dtype=dtypes.complex_dtype, device=alphas.device)
        
        # Apply linop to images
        for k in range(K):
            ksp += self.A.forward(alphas[k]) * self.phi[k]
            
        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Adjoint call of this linear model.

        Parameters
        ----------
        ksp : torch.tensor
            the k-space data with shape (C, *trj_size)
        
        Returns
        ---------
        img : torch.tensor
            the image with shape (*im_size)
        """
        K = self.phi.shape[0]
        img = torch.zeros(self.ishape, dtype=dtypes.complex_dtype, device=ksp.device)
        
        # Apply linop to kspace
        for k in range(K):
            img[k] = self.A.adjoint(ksp * self.phi[k].conj())
        
        return img

    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Gram/normal call of this linear model (A.H (A (x))).

        Parameters
        ----------
        img : torch.tensor
            the image with shape (*im_size)
        
        Returns
        ---------
        img_hat : torch.tensor
            the ouput image with shape (*im_size)
        """
        return self.adjoint(self.forward(img))
        

class subspace_linop(linop):
    """
    Linop for subspace models
    """

    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 use_toeplitz: Optional[bool] = False,
                 spatial_funcs: Optional[torch.Tensor] = None,
                 temporal_funcs: Optional[torch.Tensor] = None,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (nro, npe, ntr, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        phi : torch.tensor <complex> | GPU
            subspace basis with shape (nsub, ntr)
        mps : torch.tensor <complex> | GPU
            sensititvity maps with shape (ncoil, ndim1, ..., ndimN)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (nro, ...)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        use_toeplitz : bool
            toggles toeplitz normal operator
        spatial_funcs : torch.tensor <complex> | GPU
            the spatial functions for imperfection models with shape (L, *im_size)
        temporal_funcs : torch.tensor <complex> | GPU
            the temporal functions for imperfection models with shape (L, *trj_size)
        bparams : batching_params
            contains the batch sizes for the coils, subspace coeffs, and field segments
        """
        ishape = (phi.shape[0], *im_size)
        oshape = (mps.shape[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)

        # Consts
        torch_dev = trj.device
        assert phi.device == torch_dev
        assert mps.device == torch_dev

        # Default params
        if nufft is None:
            nufft = torchkb_nufft(im_size, torch_dev.index)
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=dtypes.real_dtype, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Rescale and type cast
        trj = nufft.rescale_trajectory(trj).type(dtypes.real_dtype)
        dcf = dcf.type(dtypes.real_dtype)
        mps = mps.type(dtypes.complex_dtype)
        phi = phi.type(dtypes.complex_dtype)
        
        # Compute toeplitz kernels
        if use_toeplitz:

            # Weighting functions
            phis = phi.conj()[:, None, :] * phi[None, ...] # nsub nsub ntr
            weights = rearrange(phis, 'nsub1 nsub2 ntr -> (nsub1 nsub2) 1 1 ntr')
            weights = weights * dcf

            if spatial_funcs is not None:
                raise NotImplementedError

            # Compute kernels
            toep_kerns = None
            for a, b in batch_iterator(weights.shape[0], batching_params.toeplitz_batch_size):
                kerns = nufft.calc_teoplitz_kernels(trj[None,], weights[None, a:b])[0]
                if toep_kerns is None:
                    toep_kerns = torch.zeros((weights.shape[0], *kerns.shape[1:]), dtype=dtypes.complex_dtype, device=torch_dev)
                toep_kerns[a:b] = kerns

            # Reshape 
            self.toep_kerns = rearrange(toep_kerns, '(nsub1 nsub2) ... -> nsub1 nsub2 ...',
                                        nsub1=phi.shape[0], nsub2=phi.shape[0])
        else:
            self.toep_kerns = None
        
        if spatial_funcs is not None:
            assert temporal_funcs is not None
            assert spatial_funcs.shape[1:] == im_size
            assert temporal_funcs.shape[1:] == trj.shape[:-1]
            self.spatial_funcs = spatial_funcs
            self.temporal_funcs = temporal_funcs
        else:
            spatial_funcs = torch.ones((1,)*(len(im_size)+1), dtype=dtypes.complex_dtype, device=torch_dev)
            spatial_funcs = torch.ones((1,)*(len(dcf)+1), dtype=dtypes.complex_dtype, device=torch_dev)

        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.trj = trj
        self.phi = phi
        self.mps = mps
        self.dcf = dcf
        self.nufft = nufft
        self.bparams = bparams
        self.torch_dev = torch_dev

    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        alphas : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, *im_size)
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, nro, npe, ntr)
        """

        # Useful constants
        imperf_rank = self.spatial_funcs.shape[0]
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        sub_batch_size = self.bparams.sub_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((nc, *self.trj.shape[:-1]), dtype=dtypes.complex_dtype, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]

            # Batch over segments
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                
                # Feild correction
                mps_weighted = mps[:, None] * self.spatial_funcs[l1:l2] # C L *im_size

                # Batch over subspace
                for a, b in batch_iterator(nsub, sub_batch_size):
                    Sx = einsum(mps_weighted, alphas[a:b], 'nc nsub nseg ..., nsub ... -> nc nsub nseg ...')

                    # NUFFT and phi
                    FSx = self.nufft.forward(Sx[None,], self.trj[None, ...])[0]

                    # Subspace
                    PFSx = einsum(FSx, self.phi[a:b], 'nc nsub nseg nro npe ntr, nsub ntr -> nc nseg nro npe ntr')

                    # Field correction
                    PFSx = (PFSx * self.temporal_funcs[l1:l2]).sum(dim=1) # C ... 
                    
                    # Append to k-space
                    ksp[c:d, ...] += PFSx

        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Adjoint call of this linear model.

        Parameters
        ----------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, nro, npe, ntr)
        
        Returns
        ---------
        alphas : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, *im_size)
        """

        # Useful constants
        imperf_rank = self.imperf_rank
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        sub_batch_size = self.bparams.sub_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result subspace coefficients
        alphas = torch.zeros((nsub, *self.im_size), dtype=dtypes.complex_dtype, device=self.torch_dev)
        
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            ksp_weighted = ksp[c:d, ...] * self.dcf[None, ...]

            # Batch over segments
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                
                # Feild correction
                Wy = ksp_weighted[:, None] * self.temporal_funcs[l1:l2].conj() # C L *trj_size

                # Batch over subspace
                for a, b in batch_iterator(nsub, sub_batch_size):
                    PWy = einsum(Wy, self.phi.conj()[a:b], 'nc nseg nro npe ntr, nsub ntr -> nc nsub nseg nro npe ntr')
                    FPWy = self.nufft.adjoint(PWy[None, ...], self.trj[None, ...])[0] # nc nsub nseg *im_size

                    # Conjugate maps
                    SFPWy = einsum(FPWy, mps.conj(), 'nc nsub nseg ..., nc ... -> nsub nseg ...')

                    # Conjugate imperfection maps
                    SFPWy = (SFPWy * self.spatial_funcs[l1:l2].conj()).sum(dim=1) # K *im_size

                    # Append to image
                    alphas[a:b, ...] += SFPWy

        return alphas
    
    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        """
        Gram or normal call of this linear model (A.H (A (x))).

        Parameters
        ----------
        alphas : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, *im_size)
        
        Returns
        ---------
        alphas_hat : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, *im_size)
        """
        
        # Do forward and adjoint, a bit slow
        if not self.use_toeplitz:
            return self.adjoint(self.forward(alphas))
        else:

            # Useful constants
            nsub = self.phi.shape[0]
            nc = self.mps.shape[0]
            dim = len(self.im_size)
            coil_batch_size = self.bparams.coil_batch_size
            sub_batch_size = self.bparams.sub_batch_size

            # Padding operator
            im_size_os = self.toep_kerns.shape[-dim:]
            padder = PadLast(im_size_os, self.im_size)

            # Result array
            alphas_hat = torch.zeros_like(alphas)
                    
            # Batch over coils
            for c, d in batch_iterator(nc, coil_batch_size):
                mps = self.mps[c:d]

                # Batch over subspace
                for a, b in batch_iterator(nsub, sub_batch_size):
                    alpha = alphas[a:b]

                    # Apply Coils and FT
                    Sx = mps[:, None, ...] * alpha[None, ...] 
                    RSx = padder.forward(Sx)
                    FRSx = fft(RSx, dim=tuple(range(-dim, 0))) # nc nsub *im_size_os

                    # Apply Toeplitz kernels
                    for i in range(nsub):
                        kerns = self.toep_kerns[i, a:b, ...]
                        MFBSx = einsum(kerns, FRSx, 'nsub ... , nc nsub ... -> nc ...')
                        FMFBSx = ifft(MFBSx, dim=tuple(range(-dim, 0))) 
                        RFMFBSx = padder.adjoint(FMFBSx) 

                        # Apply adjoint mps
                        SRFMFBSx = einsum(RFMFBSx, mps.conj(), 'nc ... , nc ... -> ...')
                        
                        # Update output
                        alphas_hat[i] += SRFMFBSx
        
        return alphas_hat
        