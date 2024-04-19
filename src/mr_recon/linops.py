import gc
import torch
import torch.nn as nn

from dataclasses import dataclass
from mr_recon.fourier import fft, ifft
from mr_recon.utils import batch_iterator
from mr_recon.pad import PadLast
from mr_recon.imperfections.imperfection import imperfection
from mr_recon.fourier import (
    gridded_nufft,
    sigpy_nufft,
    torchkb_nufft,
    NUFFT
)
from mr_recon.indexing import multi_grid
from einops import rearrange, einsum
from typing import Optional
from tqdm import tqdm

""""
In the comments and code, we use a few acronyms/shortened nouns. 
Here is a list and description of each:
    nx   - number of pixels in the x-direction
    ny   - number of pixels in the y-direction
    nz   - number of pixels in the z-direction
    nc   - number of coil sensitivity maps
    nseg - number of B0 time segments
    nro  - number of points along the readout dimenstion
    npe  - number of phase encodes/groups/interleaves
    ntr  - number of TRs (time repetition)
    nsub - number of subspace coeffs
    d    - dimension of the problem. d = 2 for 2D, d = 3 for 3D, etc
"""

@dataclass
class batching_params:
    coil_batch_size: Optional[int] = 1
    sub_batch_size: Optional[int] = 1
    field_batch_size: Optional[int] = 1
    toeplitz_batch_size: Optional[int] = 1

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

class sense_linop(linop):
    """
    Linop for sense models
    """
    
    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 imperf_model: Optional[imperfection] = None,
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (*trj_size, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        mps : torch.tensor <complex> | GPU
            sensititvity maps with shape (ncoil, *im_size)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (*trj_size)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        imperf_model : lowdim_imperfection
            models imperfections with lowrank splitting
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains various batch sizes
        """
        ishape = im_size
        oshape = (mps.shape[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)

        # Consts
        torch_dev = trj.device
        assert mps.device == torch_dev

        # Default params
        if nufft is None:
            nufft = torchkb_nufft(im_size, torch_dev.index)
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Rescale and change types
        trj = nufft.rescale_trajectory(trj).type(torch.float32)
        dcf = dcf.type(torch.float32)
        mps = mps.type(torch.complex64)
        
        # Compute toeplitz kernels
        if use_toeplitz:
            if imperf_model is None:
                self.toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], dcf[None,])[0]
            else:
                # tfs = imperf_model.temporal_funcs
                y_ones = torch.ones(trj.shape[:-1], dtype=torch.complex64, device=torch_dev)
                tfs = imperf_model.apply_temporal_adjoint(y_ones).conj()
                weights = einsum(tfs.conj(), tfs, 'nseg1 ... , nseg2 ... -> nseg1 nseg2 ...')
                weights = rearrange(weights, 'n1 n2 ... -> (n1 n2) ... ') * dcf[None, ...]
                self.toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], weights) 
                self.toep_kerns = rearrange(self.toep_kerns, '(n1 n2) ... -> n1 n2 ...', n1=tfs.shape[0], n2=tfs.shape[0])
        else:
            self.toep_kerns = None

        if imperf_model is not None:
            self.imperf_rank = imperf_model.L
        else:
            self.imperf_rank = 1

        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.trj = trj
        self.mps = mps
        self.dcf = dcf
        self.nufft = nufft
        self.bparams = bparams
        self.imperf_model = imperf_model
        self.torch_dev = torch_dev

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        alphas : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        """

        # Useful constants
        imperf_rank = self.imperf_rank
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((nc, *self.trj.shape[:-1]), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps_times_img = self.mps[c:d] * img

            # Batch over segments 
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                if self.imperf_model is None:
                    Sx = mps_times_img[:, None, ...]
                else:
                    Sx = self.imperf_model.apply_spatial(mps_times_img, slice(l1, l2))

                # NUFFT and temporal terms
                FSx = self.nufft.forward(Sx[None,], self.trj[None, ...])[0]

                if self.imperf_model is None:
                    HFSx = FSx[:, 0, ...]
                else:
                    HFSx = self.imperf_model.apply_temporal(FSx, slice(l1, l2))
                
                # Append to k-space
                ksp[c:d, ...] += HFSx

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
        imperf_rank = self.imperf_rank
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result image
        img = torch.zeros(self.im_size, dtype=torch.complex64, device=self.torch_dev)
            
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            ksp_weighted = ksp[c:d, ...] * self.dcf[None, ...]

            # Batch over segments
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                if self.imperf_model is None:
                    HWy = ksp_weighted[:, None, ...]
                else:
                    HWy = self.imperf_model.apply_temporal_adjoint(ksp_weighted, slice(l1, l2))
                
                # Adjoint
                FHWy = self.nufft.adjoint(HWy[None, ...], self.trj[None, ...])[0] # nc nseg *im_size

                # Conjugate maps
                SFHWy = einsum(FHWy, mps.conj(), 'nc nseg ..., nc ... -> nseg ...')

                if self.imperf_model is None:
                    BSFHWy = SFHWy[0]
                else:
                    BSFHWy = self.imperf_model.apply_spatial_adjoint(SFHWy, slice(l1, l2))

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
            imperf_rank = self.imperf_rank
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

                # Batch over segments
                for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                    # Apply Coils and spatial funcs
                    if self.imperf_model is None:
                        mps_weighted = mps[:, None, ...]
                    else:
                        mps_weighted = self.imperf_model.apply_spatial(mps, slice(l1, l2))
                    Sx = mps_weighted * img

                    RSx = padder.forward(Sx)
                    FRSx = fft(RSx, dim=tuple(range(-dim, 0))) # nc nseg *im_size_os

                    # Apply Toeplitz kernels
                    if self.imperf_model is None:
                        MFBSx = self.toep_kerns * FRSx[:, 0, ...]
                        FMFBSx = ifft(MFBSx, dim=tuple(range(-dim, 0))) 
                        RFMFBSx = padder.adjoint(FMFBSx) 
                    else:
                        MFBSx = einsum(self.toep_kerns[:, l1:l2, ...],  FRSx,
                                       'nseg nseg2 ..., nc nseg2 ... -> nc nseg ...')
                        FMFBSx = ifft(MFBSx, dim=tuple(range(-dim, 0))) 
                        RFMFBSx = padder.adjoint(FMFBSx)
                        RFMFBSx = self.imperf_model.apply_spatial_adjoint(RFMFBSx) # Batch?

                    # Apply adjoint mps
                    SRFMFBSx = einsum(RFMFBSx, mps.conj(), 'nc ... , nc ... -> ...')
                    
                    # Update output
                    img_hat += SRFMFBSx
        
        return img_hat
        
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
                 imperf_model: Optional[imperfection] = None,
                 use_toeplitz: Optional[bool] = False,
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
        imperf_model : lowdim_imperfection
            models imperfections with lowrank splitting
        use_toeplitz : bool
            toggles toeplitz normal operator
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
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Rescale and type cast
        trj = nufft.rescale_trajectory(trj).type(torch.float32)
        dcf = dcf.type(torch.float32)
        mps = mps.type(torch.complex64)
        phi = phi.type(torch.complex64)
        
        # Compute toeplitz kernels
        if use_toeplitz:

            # Weighting functions
            phis = phi.conj()[:, None, :] * phi[None, ...] # nsub nsub ntr
            weights = rearrange(phis, 'nsub1 nsub2 ntr -> (nsub1 nsub2) 1 1 ntr')
            weights = weights * dcf

            if imperf_model is not None:
                raise NotImplementedError

            # Compute kernels
            toep_kerns = None
            for a, b in batch_iterator(weights.shape[0], batching_params.toeplitz_batch_size):
                kerns = nufft.calc_teoplitz_kernels(trj[None,], weights[a:b])
                if toep_kerns is None:
                    toep_kerns = torch.zeros((weights.shape[0], *kerns.shape[1:]), dtype=torch.complex64, device=torch_dev)
                toep_kerns[a:b] = kerns

            # Reshape 
            self.toep_kerns = rearrange(toep_kerns, '(nsub1 nsub2) ... -> nsub1 nsub2 ...',
                                        nsub1=phi.shape[0], nsub2=phi.shape[0])
        else:
            self.toep_kerns = None
        
        if imperf_model is not None:
            self.imperf_rank = imperf_model.L
        else:
            self.imperf_rank = 1

        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.trj = trj
        self.phi = phi
        self.mps = mps
        self.dcf = dcf
        self.nufft = nufft
        self.imperf_model = imperf_model
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
        imperf_rank = self.imperf_rank
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        sub_batch_size = self.bparams.sub_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((nc, *self.trj.shape[:-1]), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]

            # Batch over segments
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                
                # Feild correction
                if self.imperf_model is None:
                    mps_weighted = mps[:, None, None, ...]
                else:
                    mps_weighted = self.imperf_model.apply_spatial(mps[:, None, ...], slice(l1, l2))

                # Batch over subspace
                for a, b in batch_iterator(nsub, sub_batch_size):
                    Sx = einsum(mps_weighted, alphas[a:b], 'nc nsub nseg ..., nsub ... -> nc nsub nseg ...')

                    # NUFFT and phi
                    FSx = self.nufft.forward(Sx[None,], self.trj[None, ...])[0]

                    # Subspace
                    PFSx = einsum(FSx, self.phi[a:b], 'nc nsub nseg nro npe ntr, nsub ntr -> nc nseg nro npe ntr')

                    # Field correction
                    if self.imperf_model is None:
                        PFSx = PFSx[:, 0, ...]
                    else:
                        PFSx = self.imperf_model.apply_temporal(PFSx, slice(l1, l2))

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
        alphas = torch.zeros((nsub, *self.im_size), dtype=torch.complex64, device=self.torch_dev)
            
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            ksp_weighted = ksp[c:d, ...] * self.dcf[None, ...]

            # Batch over segments
            for l1, l2 in batch_iterator(imperf_rank, seg_batch_size):
                
                # Feild correction
                if self.imperf_model is None:
                    Wy = ksp_weighted[:, None, ...]
                else:
                    Wy = self.imperf_model.apply_temporal_adjoint(ksp_weighted, slice(l1, l2))

                # Batch over subspace
                for a, b in batch_iterator(nsub, sub_batch_size):
                    PWy = einsum(Wy, self.phi.conj()[a:b], 'nc nseg nro npe ntr, nsub ntr -> nc nsub nseg nro npe ntr')
                    FPWy = self.nufft.adjoint(PWy[None, ...], self.trj[None, ...])[0] # nc nsub nseg *im_size

                    # Conjugate maps
                    SFPWy = einsum(FPWy, mps.conj(), 'nc nsub nseg ..., nc ... -> nsub nseg ...')

                    # Conjugate imperfection maps
                    if self.imperf_model is None:
                        SFPWy = SFPWy[:, 0]
                    else:
                        SFPWy = self.imperf_model.apply_spatial_adjoint(SFPWy, slice(l1, l2))

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
        