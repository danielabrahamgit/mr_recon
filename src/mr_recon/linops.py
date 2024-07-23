import gc
from textwrap import indent
import torch
import torch.nn as nn

from dataclasses import dataclass
from mr_recon.fourier import fft, ifft
from mr_recon.utils import batch_iterator, gen_grd
from mr_recon.pad import PadLast
from mr_recon.imperfections.imperfection import imperfection
from mr_recon.fourier import (
    gridded_nufft,
    sigpy_nufft,
    torchkb_nufft,
    NUFFT
)
from mr_recon.multi_coil.grappa_est import train_kernels
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

class multi_chan_linop(linop):
    """
    Linop for doing channel by channel reconstruction
    """

    def __init__(self,
                 out_size: tuple,
                 trj: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 imperf_model: Optional[imperfection] = None,
                 use_toeplitz: Optional[bool] = False,
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
        imperf_model : lowdim_imperfection
            models imperfections with lowrank splitting
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains various batch sizes
        """
        ishape = out_size
        oshape = (out_size[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)

        mps_dummy = torch.ones((1, *out_size[1:]), dtype=torch.complex64, device=trj.device)
        self.A = sense_linop(out_size[1:], trj, mps_dummy, dcf, nufft, imperf_model, use_toeplitz, bparams)

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
        ksp = torch.zeros(self.oshape, dtype=torch.complex64, device=img.device)
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
        img = torch.zeros(self.ishape, dtype=torch.complex64, device=ksp.device)
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
        img_hat = torch.zeros(self.ishape, dtype=torch.complex64, device=img.device)
        for i in range(self.ishape[0]):
            img_hat[i] = self.A.normal(img[i])
        return img_hat

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
            nufft = sigpy_nufft(im_size)
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
                y_ones = torch.ones(trj.shape[:-1], dtype=torch.complex64, device=torch_dev)
                tfs = imperf_model.apply_temporal_adjoint(y_ones).conj()
                weights = einsum(tfs.conj(), tfs, 'nseg1 ... , nseg2 ... -> nseg1 nseg2 ...')
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
        img : torch.tensor <complex> | GPU
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

class sense_linop_grog(linop):
    """
    Sense linop designed for grogged data.
    """

    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 os_grid: float,
                 dcf: Optional[torch.Tensor] = None,
                 inv_noise_cov: Optional[torch.Tensor] = None,
                 imperf_model: Optional[imperfection] = None,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        trj : torch.tensor 
            The k-space trajectory with shape (*trj_size, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        mps : torch.tensor 
            sensititvity maps with shape (ncoil, *im_size)
        os_grid : float
            the grid oversampling factor for grog
        dcf : torch.tensor 
            the density comp. functon with shape (*trj_size)
        inv_noise_cov : torch.tensor
            the inverse noise covariance matrix with shape (..., nc, nc)
        imperf_model : lowdim_imperfection
            models imperfections with lowrank splitting
        bparams : batching_params
            contains various batch sizes
        """
        ishape = im_size
        oshape = (mps.shape[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)

        nufft = gridded_nufft(im_size, grid_oversamp=os_grid)
        self.A = sense_linop(im_size, trj, mps, dcf, nufft, imperf_model, False, bparams)

        self.inv_noise_cov = inv_noise_cov
    
    def set_noise_cov(self,
                      inv_noise_cov: torch.Tensor):
        self.inv_noise_cov = inv_noise_cov

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
            the k-space data with shape (nc, *trj_size)
        """
        ksp = self.A.forward(img)
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
        # Apply noise covaraince
        if self.inv_noise_cov is not None:
            ksp = einsum(ksp, self.inv_noise_cov, 'ci ..., ... co ci -> co ...')
        img = self.A.adjoint(ksp)
        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
          """
          Gram/normal call of this linear model (A.H (A (x))).
    
          Parameters
          ----------
          img : torch.tensor <complex>
                the image with shape (*im_size)
          
          Returns
          ---------
          img_hat : torch.tensor <complex>
                the ouput image with shape (*im_size)
          """
          img_hat = self.adjoint(self.forward(img))
          return img_hat

class spirit_linop(linop):
    """
    Linops for spirit models. We will use notation from miki's paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2925465/

    min_m ||Dm - y||_2^2 + lamda ||(G - I) m||_2^2

    same as solving

    (D^H D + lamda (G - I)^H (G - I)) m = D^H y

    Thus:
    self.normal = D^H D + lamda (G - I)^H (G - I)
    self.adjoint = D^H 
    self.forward = D
    self.forward_grappa = G
    """
    
    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 ksp_cal: torch.Tensor,
                 kern_size: tuple,
                 lamda: Optional[float] = 1e-5,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (*trj_size, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        ksp_cal : torch.tensor <complex> | GPU
            the calibration data with shape (nc, *cal_size)
        kern_size : tuple
            size of the kernel for the spirit model, has shape (*kern_size)
        lamda : float
            the regularization parameter for the spirit model
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
        ishape = (ksp_cal.shape[0], *im_size)
        oshape = (ksp_cal.shape[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)

        # Consts
        torch_dev = trj.device
        assert ksp_cal.device == torch_dev

        # Default params
        if nufft is None:
            nufft = sigpy_nufft(im_size)
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Rescale and change types
        trj = nufft.rescale_trajectory(trj).type(torch.float32)
        dcf = dcf.type(torch.float32)
        
        # Save
        self.lamda = lamda
        self.im_size = im_size
        self.trj = trj
        self.dcf = dcf
        self.nufft = nufft
        self.bparams = bparams
        self.torch_dev = torch_dev
        
class grappa_linop(linop):
    """
    Linop for applying cartesian grappa kernels.
    """

    def __init__(self,
                 im_size: tuple,
                 ksp_cal: torch.Tensor,
                 kern_size: tuple,
                 image_domain: Optional[bool] = True,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        ksp_cal : torch.tensor <complex> | GPU
            the calibration data with shape (nc, *cal_size)
        kern_size : tuple
            size of the kernel for the spirit model, has shape (*kern_size)
        image_domain : bool
            If true, applies kernel to images. If false, applies to k-space.
        bparams : batching_params
            contains various batch sizes
        """
        ishape = (ksp_cal.shape[0], *im_size)
        oshape = (ksp_cal.shape[0], *im_size)
        super().__init__(ishape, oshape)

        # Consts
        torch_dev = ksp_cal.device
        assert ksp_cal.device == torch_dev
        assert len(im_size) == len(kern_size)

        # Train kernels
        img_cal = ifft(ksp_cal, dim=tuple(range(-len(im_size), 0)))
        source_vecs = gen_grd(kern_size, kern_size).reshape((1, -1, len(kern_size))).to(torch_dev)
        kernel = train_kernels(img_cal, source_vecs, lamda_tikonov=1e-3)[0]
        
        # Reshape/transform kernels
        if image_domain:
            device_idx = torch_dev.index
            if 'cpu' in str(torch_dev).lower():
                device_idx = -1
            nfft = torchkb_nufft(im_size, device_idx)
            kernel = nfft.adjoint(kernel[None,], source_vecs[None,])[0]
        else:
            kernel = kernel.reshape((*kernel.shape[:-1], *kern_size))
        breakpoint()


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
        