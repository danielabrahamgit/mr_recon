import gc
import torch
import torch.nn as nn

from dataclasses import dataclass
from mr_recon.fourier import fft, ifft
from mr_recon.utils import batch_iterator
from mr_recon.pad import PadLast
from mr_recon.imperfections.field import field_handler
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
        
        # Compute toeplitz kernels
        if use_toeplitz:
            self.toep_kerns = nufft.calc_teoplitz_kernels(trj[None,], dcf[None,])[0]
        else:
            self.toep_kerns = None

        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.trj = trj
        self.mps = mps.type(torch.complex64)
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
        alphas : torch.tensor <complex> | GPU
            the image with shape (*im_size)
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, *trj_size)
        """

        # Useful constants
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size

        # Result array
        ksp = torch.zeros((nc, *self.trj.shape[:-1]), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            Sx = mps * img

            # NUFFT and phi
            FSx = self.nufft.forward(Sx[None,], self.trj[None, ...])[0]

            # Append to k-space
            ksp[c:d, ...] += FSx

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
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size

        # Result subspace coefficients
        img = torch.zeros(self.im_size, dtype=torch.complex64, device=self.torch_dev)
            
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            Wy = ksp[c:d, ...] * self.dcf
            
            # Adjoint
            FWy = self.nufft.adjoint(Wy[None, ...], self.trj[None, ...])[0] # nc *im_size

            # Conjugate maps
            SFWy = einsum(FWy, mps.conj(), 'nc ..., nc ... -> ...')

            # Append to image
            img += SFWy

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
            nc = self.mps.shape[0]
            dim = len(self.im_size)
            coil_batch_size = self.bparams.coil_batch_size

            # Padding operator
            im_size_os = self.toep_kerns.shape[-dim:]
            padder = PadLast(im_size_os, self.im_size)

            # Result array
            img_hat = torch.zeros_like(img)
                    
            # Batch over coils
            for c, d in batch_iterator(nc, coil_batch_size):
                mps = self.mps[c:d]

                # Apply Coils and FT
                Sx = mps * img
                RSx = padder.forward(Sx)
                FRSx = fft(RSx, dim=tuple(range(-dim, 0))) # nc *im_size_os

                # Apply Toeplitz kernels
                MFBSx = self.toep_kerns * FRSx
                FMFBSx = ifft(MFBSx, dim=tuple(range(-dim, 0))) 
                RFMFBSx = padder.adjoint(FMFBSx) 

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
        
        # Rescale trajectory to match nufft format
        trj = nufft.rescale_trajectory(trj).type(torch.float32)
        
        # Compute toeplitz kernels
        if use_toeplitz:

            # Weighting functions
            phis = phi.conj()[:, None, :] * phi[None, ...] # nsub nsub ntr
            weights = rearrange(phis, 'nsub1 nsub2 ntr -> (nsub1 nsub2) 1 1 ntr')
            weights = weights * dcf

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

        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.trj = trj
        self.phi = phi.type(torch.complex64)
        self.mps = mps.type(torch.complex64)
        self.dcf = dcf.type(torch.float32)
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
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        sub_batch_size = self.bparams.sub_batch_size

        # Result array
        ksp = torch.zeros((nc, *self.trj.shape[:-1]), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]

            # Batch over subspace
            for a, b in batch_iterator(nsub, sub_batch_size):
                alpha = alphas[a:b]
                Sx = mps[:, None, ...] * alpha[None, ...]

                # NUFFT and phi
                FSx = self.nufft.forward(Sx[None,], self.trj[None, ...])[0]
                PFSx = einsum(FSx, self.phi[a:b], 'nc nsub nro npe ntr, nsub ntr -> nc nro npe ntr')

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
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size
        sub_batch_size = self.bparams.sub_batch_size

        # Result subspace coefficients
        alphas = torch.zeros((nsub, *self.im_size), dtype=torch.complex64, device=self.torch_dev)
            
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            Wy = ksp[c:d, ...] * self.dcf[None, ...] # nc nro npe ntr
            
            # Batch over subspace
            for a, b in batch_iterator(nsub, sub_batch_size):
                PWy = einsum(Wy, self.phi.conj()[a:b], 'nc nro npe ntr, nsub ntr -> nsub nc nro npe ntr')
                FPWy = self.nufft.adjoint(PWy[None, ...], self.trj[None, ...])[0] # nsub nc *im_size

                # Conjugate maps
                SFPWy = einsum(FPWy, mps.conj(), 'nsub nc ..., nc ... -> nsub ...')

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
        
class subspace_linop_old(nn.Module):
    
    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 field_obj: Optional[field_handler] = None,
                 use_toeplitz: Optional[bool] = True,
                 grog_grid_oversamp: Optional[float] = None,
                 coil_batch_size: Optional[int] = 1,
                 sub_batch_size: Optional[int] = 1,
                 seg_batch_size: Optional[int] = 1):
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
        field_obj : field_handler
            represents phase due to field imperfections
        use_toeplitz : bool
            toggles toeplitz normal operator
        grog_grid_oversamp : float 
            If given, toggles Gridded recon. Assumes trj lines on oversampled grid with 
            oversampling factor 'grog_grid_oversamp' (usually between 1 and 2)
        coil_batch_size : int
            number of coils to batch on gpu
        sub_batch_size : int
            number of subspace coeffs to batch on gpu
        seg_batch_size : int
            number of time segments to batch on gpu
        """
        super().__init__()

        # Useful constants
        nsub, ntr = phi.shape
        self.im_size = im_size
        self.alpha_size = (nsub, *self.im_size)
        self.ishape = (nsub, *self.im_size)
        self.oshape = (mps.shape[0], *trj.shape[:-1])
        for i in im_size:
            assert i % 2 == 0, 'Only supports even dims'

        # Store torch device
        self.torch_dev = trj.device
        assert phi.device == self.torch_dev

        # Make self.nufft
        device_idx = trj.get_device()
        if grog_grid_oversamp is not None:
            self.nufft = gridded_nufft(im_size, device_idx, grog_grid_oversamp)
        else:
            # self.nufft = torchkb_nufft(im_size, device_idx)
            self.nufft = sigpy_nufft(im_size, device_idx)
        trj = self.nufft.rescale_trajectory(trj)

        # Default dcf
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.complex64).to(self.torch_dev)
        
        # Save
        self.trj = trj
        self.dcf = dcf
        self.field_obj = field_obj
        self.nseg = 1 if field_obj is None else field_obj.nseg
        self.phi = phi
        self.mps = mps
        self.coil_batch_size = coil_batch_size
        self.sub_batch_size = sub_batch_size
        self.seg_batch_size = seg_batch_size
        self.grog_grid_oversamp = grog_grid_oversamp
        self.use_toeplitz = use_toeplitz

        # Toeplitz kernels
        if use_toeplitz:
            
            if grog_grid_oversamp:
                self.toep_kerns = self._compute_grog_toeplitz_kernels()
            else:
                self.toep_kerns = self._compute_toeplitz_kernels()
            
            # Cleanup
            gc.collect()
            if 'cpu' not in str(self.torch_dev):
                with torch.cuda.device(self.torch_dev):
                    torch.cuda.empty_cache()

    def update_trj(self, 
                   trj: torch.Tensor,
                   dcf: Optional[torch.Tensor] = None):
        """
        Updates trajectory.

        Parameters:
        -----------
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (nro, npe, ntr, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (nro, ...)
        
        Saves/Updates:
        --------------
        self.trj, self.dcf
        """

        assert trj.device == self.trj.device
        if dcf is not None:
            assert trj.shape[:-1] == dcf.shape
            assert trj.device == dcf.device
        self.trj = self.nufft.rescale_trajectory(trj)

        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.complex64).to(self.torch_dev)
        self.dcf = dcf

        if self.use_toeplitz:
            if self.grog_grid_oversamp:
                self.toep_kerns = self._compute_grog_toeplitz_kernels()
            else:
                self.toep_kerns = self._compute_toeplitz_kernels()
            
            # Cleanup
            gc.collect()
            if 'cpu' not in str(self.torch_dev):
                with torch.cuda.device(self.torch_dev):
                    torch.cuda.empty_cache()

    def _compute_grog_toeplitz_kernels(self) -> torch.Tensor:
        """
        Computes toeplitz kernels for grog/cartesian subspace recon.

        Returns:
        ---------
        Ts : torch.tensor <complex>
            The subspace toeplitz kernels with shape (nsub, nsub, nx, ny, ...)
            No oversampling necessary!
        """

        if self.field_obj is not None:
            raise NotImplementedError

        # Useful constants
        nsub = self.phi.shape[0]
        nseg, nro, npe, ntr, d = self.trj.shape
        phi = self.phi
        dcf = self.dcf
        
        # grog Toeplitz kernels
        im_size_os = tuple([int(round(i * self.grog_grid_oversamp)) for i in self.im_size])
        Ts_grog = torch.zeros((nseg, nsub, nsub, *im_size_os), dtype=torch.complex64)

        # Compute grog topelitz kernels
        for t, u in batch_iterator(nseg, self.seg_batch_size):

            # Get trj and dcf for this segment
            trj_seg = self.trj[t:u]
            dcf_seg = dcf[t:u]

            # Do adjoint in batches
            for a1, b1 in batch_iterator(nsub, self.sub_batch_size):
                for a2, b2 in batch_iterator(nsub, self.sub_batch_size):
                    data_pts = phi[None, a1:b1, None, None, None, :].conj() * phi[None, None, a2:b2, None, None, :] * dcf_seg[:, None, None, ...]
                    # Ts_grog[:, a1:b1, a2:b2] += sp_ifft(self.nufft.adjoint(data_pts, trj_seg), dim=tuple(range(-d, 0))).cpu()
                    for i in range(t, u):
                        i_zero_start = i - t
                        Ts_grog[i, a1:b1, a2:b2] += multi_grid(data_pts[i_zero_start], trj_seg[i_zero_start], final_size=im_size_os).cpu()

        return Ts_grog 
                         
    def _compute_toeplitz_kernels(self,
                                  os_factor: Optional[float] = 2.0) -> torch.Tensor:
        """
        Computes toeplitz kernels for subspace recon.

        Parameters:
        -----------
        os_factor : float
            oversamp factor for toeplitz

        Returns:
        ---------
        Ts : torch.tensor <complex>
            The subspace toeplitz kernels with shape (nsub, nsub, os_factor * nx, os_factor * ny, ...)
        """

        if self.field_obj is not None:
            raise NotImplementedError
        
        # Useful constants
        nsub = self.phi.shape[0]
        nseg, nro, npe, ntr, d = self.trj.shape
        data_dim = (nro, npe, ntr)

        # Toeplitz kernels
        im_size_os = torch.round(torch.tensor(self.im_size) * os_factor).type(torch.int).tolist()
        Ts = torch.zeros((nseg, nsub, nsub, *im_size_os), dtype=torch.complex64)

        # Make oversampled adjoint nufft
        if 'cpu' in str(self.torch_dev):
            device_idx = -1
        else:
            device_idx = self.torch_dev.index
        if isinstance(self.nufft, torchkb_nufft):
            nufft_toep = torchkb_nufft(im_size_os, device_idx)
            nufft_adjoint = lambda k, t : nufft_toep.adjoint(k, t)
        else:
            nufft_toep = sigpy_nufft(im_size_os, device_idx)
            nufft_adjoint = lambda k, t : nufft_toep.adjoint(k, t * os_factor)

        # Batch over time segments
        for l1 in tqdm(range(0, nseg, self.seg_batch_size), 'Computing Topelitz Kernels', disable=nseg == 1):
            l2 = min(l1 + self.seg_batch_size, nseg)

            # Trajectory segment
            trj_seg = self.trj[l1:l2]

            # Iterate through each column of toeplitz tensor
            for k in tqdm(range(nsub), 'Computing Toeplitz Kernels', disable=nseg > 1):

                # Set kth coeff to delta, which is all ones in freq domain
                alpha_ksp = torch.zeros((l2-l1, nsub, *data_dim), dtype=torch.complex64).to(self.torch_dev)
                alpha_ksp[:, k, ...] = 1.0

                # Transform with PHI
                sig_ksp = einsum(alpha_ksp, self.phi, 'b_seg nsub nro npe ntr, nsub ntr -> b_seg nro npe ntr')

                # Now do adjoint, batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    # Call adjoint
                    alpha_ksp = sig_ksp[:, None, ...] * self.phi.conj()[None, a:b, None, None, :]
                    alpha_ksp = alpha_ksp * self.dcf[l1:l2, None, ...]
                    with torch.cuda.device(device_idx):                        
                        psf_col = nufft_adjoint(alpha_ksp, trj_seg)

                        # Transform to k-space
                        T_col = fft(psf_col, dim=tuple(range(-d, 0))) 
                        T_col = T_col * (os_factor ** d) # Scaling
                        
                        # Update Toeplitz kernels
                        Ts[l1:l2, a:b, k, ...] += T_col.to('cpu')

                    # Clean up, these are massive operations
                    del T_col, alpha_ksp, psf_col
                    gc.collect()
                    with torch.cuda.device(device_idx):
                        torch.cuda.empty_cache()
                
        return Ts                

    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        alphas : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, nx, ny, (nz))
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, nro, npe, ntr)
        """

        # Useful constants
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        nseg = self.nseg

        # Result array
        ksp = torch.zeros((nc, *self.trj.shape[:-1]), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, self.coil_batch_size):

            # Move maps to GPU
            mps_gpu = self.mps[c:d]

            # Batch over segments
            for t, u in batch_iterator(nseg, self.seg_batch_size):

                if nseg > 1:
                    segs = slice(t, u)
                    spatial_funcs = self.field_obj.get_spatial_funcs(segs)

                # Batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    alphas_gpu = alphas[a:b]
                    Sx = mps_gpu[:, None, ...] * alphas_gpu[None, ...]

                    # Apply B0 phase
                    if nseg > 1:
                        BSx = Sx[None, ...] * spatial_funcs[:, None, None, ...]
                    else:
                        BSx = Sx[None, ...]

                    # NUFFT + phi
                    FBSx = self.nufft.forward(BSx[None,], self.trj[None, ...])[0]
                    PBFSx = einsum(FBSx, self.phi[a:b], 'nseg nc nsub nro npe ntr, nsub ntr -> nseg nc nro npe ntr')
    
                    # Apply temporal interpolators
                    if nseg > 1:
                        temporal_funcs = self.field_obj.get_temporal_funcs(segs) # nro npe ntr nseg
                        ksp_batch = einsum(temporal_funcs, PBFSx, 'nro npe ntr nseg, nseg nc nro npe ntr -> nc nro npe ntr')
                    else:
                        ksp_batch = PBFSx[0]

                    # Append to k-space
                    ksp[c:d, ...] += ksp_batch

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
            the subspace coefficient volumes with shape (nsub, nx, ny, (nz))
        """

        # Useful constants
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        nseg = self.nseg

        # Result subspace coefficients
        alphas = torch.zeros((nsub, *self.mps.shape[1:]), dtype=torch.complex64, device=self.torch_dev)
        
        # Batch over segments
        for t, u in batch_iterator(nseg, self.seg_batch_size):
            
            if nseg > 1:
                segs = slice(t, u)
                spatial_funcs = self.field_obj.get_spatial_funcs(segs)
            
            # Batch over coils
            for c, d in batch_iterator(nc, self.coil_batch_size):

                mps_gpu = self.mps[c:d]
                
                # Batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    # Adjoint phi and nufft
                    Wy = ksp[c:d, ...] * self.dcf[None, ...] # nc nro npe ntr

                    if nseg > 1:
                        temporal_funcs = self.field_obj.get_temporal_funcs(segs) # nro npe ntr nseg
                        Wy = einsum(Wy, temporal_funcs.conj(), 'nc nro npe ntr, nro npe ntr nseg -> nseg nc nro npe ntr')
                        PWy = einsum(Wy, self.phi.conj()[a:b], 'nseg nc nro npe ntr, nsub ntr -> nseg nsub nc nro npe ntr')
                    else:
                        PWy = einsum(Wy, self.phi.conj()[a:b], 'nc nro npe ntr, nsub ntr -> nsub nc nro npe ntr')
                    FPWy = self.nufft.adjoint(PWy[None, ...], self.trj[None, ...])[0] # (nseg) nsub nc *im_size

                    # Combine with adjoint maps
                    if nseg > 1:
                        BFPWy = einsum(FPWy, spatial_funcs.conj(), 'nseg nsub nc ..., nseg ... -> nsub nc ...')
                    else:
                        BFPWy = FPWy
                    SBFPWy = einsum(BFPWy, mps_gpu.conj(), 'nsub nc ..., nc ... -> nsub ...')

                    # Append to image
                    alphas[a:b, ...] += SBFPWy

        return alphas
    
    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        """
        Gram or normal call of this linear model (A.H (A (x))).

        Parameters
        ----------
        alphas : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, nx, ny, (nz))
        
        Returns
        ---------
        alphas_hat : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, nx, ny, (nz))
        """

        # TODO make this a parameter?
        save_memory = True
        
        # Do forward and adjoint, a bit slow
        if not self.use_toeplitz:
            return self.adjoint(self.forward(alphas))
        else:
            # Useful constants
            nsub = self.phi.shape[0]
            nc = self.mps.shape[0]
            nseg, nro, npe, ntr, dim = self.trj.shape

            if nseg > 1 and self.phase_mps is not None:
                self.phase_mps = self.phase_mps.to(alphas.device)
            # For padding later
            im_size_os = self.toep_kerns.shape[-dim:]
            padder = PadLast(im_size_os, self.im_size)

            # Result array
            alphas_hat = torch.zeros_like(alphas)

            # Batch over b0 time segments
            for t, u in batch_iterator(nseg, self.seg_batch_size):

                if nseg > 1:
                    phase_mps_gpu = self.phase_mps[t:u]#.to(self.torch_dev)

                # Batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    alphas_gpu = alphas[a:b]
                    if not save_memory:
                        toep_kerns = self.toep_kerns[t:u, :, a:b].to(self.torch_dev)

                    # Batch over coils
                    for c, d in batch_iterator(nc, self.coil_batch_size):
                        
                        # Apply sense/phase maps
                        mps_gpu = self.mps[c:d]
                        Sx = mps_gpu[:, None, ...] * alphas_gpu[None, ...]
                        if nseg > 1:
                            BSx = Sx[None, ...] * phase_mps_gpu[:, None, None, ...]
                        else:
                            BSx = Sx[None, ...]

                        # Toeplitz
                        BSx_os = padder.forward(BSx)
                        FBSx_os = fft(BSx_os, dim=tuple(range(-dim, 0)))
                        for a2, b2 in batch_iterator(nsub, self.sub_batch_size):
                            if save_memory:
                                kerns = self.toep_kerns[t:u, None, a2:b2, a:b].to(self.torch_dev)
                            else:
                                kerns = toep_kerns[:, None, a2:b2, :]
                            MFBSx_os = torch.sum(kerns * FBSx_os[:, :, None, ...], dim=3)
                            FMFBSx_os = ifft(MFBSx_os, dim=tuple(range(-dim, 0)))
                            FMFBSx = padder.adjoint(FMFBSx_os)

                            # Apply adjoint sense/phase maps
                            SFMFBSx = torch.sum(FMFBSx * mps_gpu[None, :, None, ...].conj(), dim=1)
                            if nseg > 1:
                                BSFMFBSx = torch.sum(SFMFBSx * phase_mps_gpu[:, None, ...].conj(), dim=0)
                            else:
                                BSFMFBSx = SFMFBSx[0]

                            # Update output
                            alphas_hat[a2:b2] += BSFMFBSx
        
        return alphas_hat
        