import gc
import torch
import torch.nn as nn

from mr_recon.utils.func import batch_iterator, sp_fft, sp_ifft
from mr_recon.utils.pad import PadLast
from mr_recon.nufft import (
    gridded_nufft,
    sigpy_nufft,
    torchkb_nufft
)
from mr_recon.utils.indexing import multi_grid
from einops import rearrange, einsum
from typing import Optional, Union
from torchkbnufft import KbNufft, KbNufftAdjoint
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

class subspace_linop(nn.Module):
    
    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 use_toeplitz: Optional[bool] = True,
                 grog_grid_oversamp: Optional[float] = None,
                 b0_dct: Optional[dict] = None,
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
        use_toeplitz : bool
            toggles toeplitz normal operator
        grog_grid_oversamp : float 
            If given, toggles Gridded recon. Assumes trj lines on oversampled grid with 
            oversampling factor 'grog_grid_oversamp' (usually between 1 and 2)
        b0_dct : dict
            dictionary with b0 info for time segmented model -
            b0_dct = {
                'b0_map': torch.tensor
                    b0 map in Hz, same dims as image
                'dt': float
                    sampling time in seconds
                'nseg': int
                    number of segments for time segmented model
            }
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

        # Grab b0 components
        if b0_dct is not None:
            nseg = b0_dct['nseg']
            dt = b0_dct['dt']
            b0_map = b0_dct['b0_map']

            # Make b0 phase maps
            split_size = round(trj.shape[0]/nseg)
            times = torch.arange(0, trj.shape[0]) * dt
            times_split = torch.split(times, split_size, dim=0)
            nseg_actual = len(times_split)
            phase_mps = torch.zeros((nseg_actual, *self.im_size), dtype=torch.complex64)
            for i in range(nseg_actual):
                ti = torch.mean(times_split[i])
                phase_mps[i] = torch.exp(-1j * 2 * torch.pi * ti * b0_map)
            phase_mps = phase_mps.to(self.torch_dev)
        else:
            nseg = 1
            phase_mps = None

        # Default dcf
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1]).to(self.torch_dev)
        
        # Process trajectory and dcf for time segmentation
        trj_seg, dcf_seg, edge_segment_size = self._time_segment_reshaper(trj, dcf, nseg)

        # Make self.nufft
        device_idx = trj_seg.get_device()
        if grog_grid_oversamp is not None:
            self.nufft = gridded_nufft(im_size, device_idx, grog_grid_oversamp)
        else:
            # self.nufft = torchkb_nufft(im_size, device_idx)
            self.nufft = sigpy_nufft(im_size, device_idx)
        
        # trj_seg = self.nufft.rescale_trajectory(trj_seg)
        
        # Save
        self.edge_segment_size = edge_segment_size
        self.trj = trj_seg
        self.dcf = dcf_seg
        self.phase_mps = phase_mps
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

    def _time_segment_reshaper(self,
                               trj: torch.Tensor,
                               dcf: torch.Tensor,
                               nseg: int) -> Union[torch.Tensor, torch.Tensor]:
        """
        Helper funciton to process the trajectory and dcf to be time segmented

        Parameters:
        -----------
        trj : torch.tensor <float>
            The k-space trajectory with shape (nro, npe, ntr, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        dcf : torch.tensor <float> 
            the density comp. functon with shape (nro, npe, ntr)
        nseg: int
            number of segments for time segmented model

        Returns
        ----------
        trj : torch.tensor <float32>
            The segmented k-space trajectory with shape (nseg, nro_new, npe, ntr, d). 
        dcf_rs : torch.tensor <float>
            The dcf with shape (nseg, nro_new, npe, ntr)
        edge_segment_size : int
            true number of readout points for the final segment
        """
        
        # Reshape trj and dcf to support segments
        split_size = round(trj.shape[0]/nseg)
        trj_split = torch.split(trj, split_size, dim=0)
        dcf_split = torch.split(dcf, split_size, dim=0)
        nseg = len(trj_split)

        # Move split components into zero-padded tensors
        trj_rs = torch.zeros((nseg, split_size, *trj.shape[1:]), dtype=torch.float32).to(self.torch_dev)
        dcf_rs = torch.zeros((nseg, split_size, *dcf.shape[1:]), dtype=torch.float32).to(self.torch_dev)
        for i in range(nseg):
            split_size_i = trj_split[i].shape[0]
            trj_rs[i, :split_size_i] = trj_split[i]
            dcf_rs[i, :split_size_i] = dcf_split[i]
        edge_segment_size = trj_split[-1].shape[0]

        return trj_rs, dcf_rs, edge_segment_size

    def _compute_grog_toeplitz_kernels(self) -> torch.Tensor:
        """
        Computes toeplitz kernels for grog/cartesian subspace recon.

        Returns:
        ---------
        Ts : torch.tensor <complex>
            The subspace toeplitz kernels with shape (nsub, nsub, nx, ny, ...)
            No oversampling necessary!
        """

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
            omega_og = rearrange(self.omega, 'nseg d (nro npe ntr) -> nseg d nro npe ntr', 
                                 nro=nro, 
                                 npe=npe, 
                                 ntr=ntr)
            trj = rearrange(omega_og, 'nseg d nro npe ntr -> nseg nro npe ntr d')[t:u]
            dcf = dcf[t:u]

            # Do adjoint in batches
            for a1, b1 in batch_iterator(nsub, self.sub_batch_size):
                for a2, b2 in batch_iterator(nsub, self.sub_batch_size):
                    data_pts = phi[None, a1:b1, None, None, None, :].conj() * phi[None, None, a2:b2, None, None, :] * dcf[:, None, None, ...]
                    for i in range(t, u):
                        i_zero_start = i - t
                        Ts_grog[i, a1:b1, a2:b2] += multi_grid(data_pts[i_zero_start], trj[i_zero_start], final_size=im_size_os).to('cpu')

        return Ts_grog 
                         
    def _compute_toeplitz_kernels(self,
                                  os_factor: Optional[float] = 2.0,
                                  trj_batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Computes toeplitz kernels for subspace recon.

        Parameters:
        -----------
        os_factor : float
            oversamp factor for toeplitz
        trj_batch_size : int
            optional batching over trajectory

        Returns:
        ---------
        Ts : torch.tensor <complex>
            The subspace toeplitz kernels with shape (nsub, nsub, os_factor * nx, os_factor * ny, ...)
        """
        
        # Useful constants
        nsub = self.phi.shape[0]
        nseg, nro, npe, ntr, d = self.trj.shape
        data_dim = (nro, npe, ntr)

        # Defualt batch
        if trj_batch_size is None:
            trj_batch_size = nro *  ntr * npe

        # Toeplitz kernels
        im_size_os = torch.round(torch.tensor(self.im_size) * os_factor).type(torch.int).tolist()
        Ts = torch.zeros((nseg, nsub, nsub, *im_size_os), dtype=torch.complex64)

        # Make oversampled adjoint nufft
        kb_adj_ob_os = KbNufftAdjoint(im_size_os, device=self.torch_dev)

        # Calculate kernels
        # Batch over time segments
        # for l1, l2 in batch_iterator(nseg, self.seg_batch_size):
        for l1 in tqdm(range(0, nseg, self.seg_batch_size), 'Computing Topelitz Kernels', disable=nseg == 1):
            l2 = min(l1 + self.seg_batch_size, nseg)

            # Iterate through each column of toeplitz tensor
            for k in tqdm(range(nsub), 'Computing Toeplitz Kernels', disable=nseg > 1):

                # Set kth coeff to delta, which is all ones in freq domain
                alpha_ksp = torch.zeros((l2-l1, nsub, *data_dim), dtype=torch.complex64).to(self.torch_dev)
                alpha_ksp[:, k, ...] = 1.0

                # Transform with PHI
                # sig_ksp = torch.sum(alpha_ksp * self.phi[None, :, None, None, :], dim=1)
                sig_ksp = einsum(alpha_ksp, self.phi, 'b_seg nsub nro npe ntr, nsub ntr -> b_seg nro npe ntr')

                # Now do adjoint, batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    # Call adjoint
                    alpha_ksp = sig_ksp[:, None, ...] * self.phi.conj()[None, a:b, None, None, :]
                    alpha_ksp = alpha_ksp * self.dcf[l1:l2, None, ...]
                    alpha_ksp_rs = rearrange(alpha_ksp, 'nseg nsub nro npe ntr -> nseg nsub (nro npe ntr)')

                    # Batch over trajectory
                    for t1, t2 in batch_iterator(alpha_ksp_rs.shape[-1], trj_batch_size):
                        with torch.cuda.device(self.torch_dev):
                            psf_col = kb_adj_ob_os(alpha_ksp_rs[..., t1:t2], 
                                                self.omega[l1:l2, ..., t1:t2], 
                                                norm='ortho')

                            # Transform to k-space
                            T_col = sp_fft(psf_col, dim=tuple(range(-d, 0))) 
                            T_col = T_col * (os_factor ** d) # Scaling
                        
                        # Update Toeplitz kernels
                        Ts[l1:l2, a:b, k, ...] += T_col.to('cpu')

                    # Clean up, these are massive operations
                    del T_col, alpha_ksp, alpha_ksp_rs, psf_col
                    gc.collect()
                    with torch.cuda.device(self.torch_dev):
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
        nseg, nro, npe, ntr, dim = self.trj.shape
        
        # Result array
        ksp = torch.zeros((nc, nseg * nro, npe, ntr), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, self.coil_batch_size):

            # Move maps to GPU
            mps_gpu = self.mps[c:d]

            # Batch over segments
            for t, u in batch_iterator(nseg, self.seg_batch_size):
            
                # Move phase maps to gpu
                if nseg > 1:
                    phase_mps_gpu = self.phase_mps[t:u]#.to(self.torch_dev)

                # Trajectory segment
                trj_seg = self.trj[t:u]

                # Batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    # Move coeffs to GPU
                    alphas_gpu = alphas[a:b]

                    # Apply maps
                    Sx = mps_gpu[:, None, ...] * alphas_gpu[None, ...]

                    # Apply B0 phase
                    if nseg > 1:
                        BSx = Sx[None, ...] * phase_mps_gpu[:, None, None, ...]
                    else:
                        BSx = Sx[None, ...]

                    # NUFFT
                    FBSx = self.nufft.forward(BSx, trj_seg)
                    # if self.grog_grid_oversamp:
                    #     BSx = self.grog_padder.forward(BSx)
                    #     FBSx_cart = sp_fft(BSx, dim=tuple(range(-dim, 0)))
                    #     FBSx_rs = torch.zeros((u-t, d-c, b-a, self.omega.shape[-1]), dtype=torch.complex64, device=self.torch_dev)
                    #     for i in range(FBSx_rs.shape[0]):
                    #         tup = (slice(None), slice(None)) + tuple(self.omega[t+i])
                    #         FBSx_rs[i] = FBSx_cart[i][tup]
                    # else:
                    #     BSx_rs = rearrange(BSx, 'nseg nc nsub ... -> nseg (nc nsub) ...')
                    #     with torch.cuda.device(self.torch_dev):
                    #         FBSx_rs = self.kb_ob(image=BSx_rs, omega=self.omega[t:u], norm='ortho')
                    #     FBSx_rs = rearrange(FBSx_rs, 'nseg (nc nsub) ... -> nseg nc nsub ...',
                    #                         nc=d-c,
                    #                         nsub=b-a)
                    # FBSx = rearrange(FBSx_rs, 'nseg nc nsub (nro npe ntr) -> nseg nc nsub nro npe ntr', 
                    #                  nro=nro, 
                    #                  npe=npe, 
                    #                  ntr=ntr)

                    # Combine with Phi
                    PBFSx = torch.sum(FBSx * self.phi[None, None, a:b, None, None, :], dim=2)
                    PBFSx_rs = rearrange(PBFSx, 'nseg nc nro npe ntr -> nc (nseg nro) npe ntr')

                    # Append to k-space
                    ksp[c:d, nro*t:nro*u, ...] += PBFSx_rs

        # Correction term
        nro_actual = nro * (nseg-1) + self.edge_segment_size

        return ksp[:, :nro_actual, ...]
    
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
        nseg, nro_new, npe, ntr, dim = self.trj.shape

        # Result subspace coefficients
        alphas = torch.zeros((nsub, *self.mps.shape[1:]), dtype=torch.complex64, device=self.torch_dev)

        # Zero pad and reshape k-space 
        num_zeros = nro_new * nseg - ksp.shape[1]
        zeros = torch.zeros((nc, num_zeros, npe, ntr), dtype=torch.complex64, device=self.torch_dev)
        ksp_zp = torch.cat((ksp, zeros), dim=1)
        ksp_rs = rearrange(ksp_zp, 
                           pattern='nc (nseg nro_new) npe ntr -> nseg nc nro_new npe ntr', 
                           nseg=nseg, 
                           nro_new=nro_new)

        # Batch over segments
        for t, u in batch_iterator(nseg, self.seg_batch_size):
            
            # Move phase maps to GPU
            if nseg > 1:
                phase_mps_gpu = self.phase_mps[t:u]#.to(self.torch_dev)

            # Grab trj segment
            trj_seg = self.trj[t:u]
            
            # Batch over coils
            for c, d in batch_iterator(nc, self.coil_batch_size):

                # Move maps to GPU
                mps_gpu = self.mps[c:d]
                
                # Batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    # Move ksp to GPU, mult by dcf
                    ksp_gpu = ksp_rs[t:u, c:d, ...] * self.dcf[t:u, None, ...]

                    # Apply adjoint phi
                    Py = ksp_gpu[:, :, None, ...] * self.phi[None, None, a:b, None, None, :].conj()

                    # adjoint NUFFT
                    # Py (nseg nc nsub nro npe ntr)
                    # trj (nseg, nro npe ntr d)
                    FPy = self.nufft.adjoint(Py, trj_seg)

                    # Py_rs = rearrange(Py, 'nseg nc nsub nro npe ntr -> nseg (nc nsub) (nro npe ntr)')
                    # if self.grog_grid_oversamp:
                    #     omega_rs = torch.moveaxis(self.omega[t:u], -1, -2)
                    #     FPy = torch.zeros((*Py_rs.shape[:2], *self.grog_os_size), dtype=torch.complex64, device=self.torch_dev)
                    #     for i in range(FPy.shape[0]):
                    #         FPy[i] = multi_grid(Py_rs[i], omega_rs[i], self.grog_os_size)
                    #     FPy = sp_ifft(FPy, dim=tuple(range(-dim, 0)))
                    #     FPy = self.grog_padder.adjoint(FPy)
                    # else:
                    #     with torch.cuda.device(self.torch_dev):
                    #         FPy = self.kb_adj_ob(data=Py_rs, omega=self.omega[t:u], norm='ortho')
                    # FPy = rearrange(FPy, f'nseg (nc nsub) ... -> nseg nc nsub ...', nc=d-c, nsub=b-a)

                    # Combine with adjoint phase arrays
                    if nseg > 1:
                        BFPy = torch.sum(FPy * phase_mps_gpu[:, None, None, ...].conj(), dim=0)
                    else:
                        BFPy = FPy[0]

                    # Combine with adjoint maps
                    SBFPy = torch.sum(BFPy * mps_gpu[:, None, ...].conj(), dim=0)

                    # Append to image
                    alphas[a:b, ...] += SBFPy

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

                # Move phase maps to GPU
                if nseg > 1:
                    phase_mps_gpu = self.phase_mps[t:u]#.to(self.torch_dev)

                # Batch over subspace
                for a, b in batch_iterator(nsub, self.sub_batch_size):

                    # Move coeffs to GPU
                    alphas_gpu = alphas[a:b]

                    if not save_memory:
                        toep_kerns = self.toep_kerns[t:u, :, a:b].to(self.torch_dev)

                    # Batch over coils
                    for c, d in batch_iterator(nc, self.coil_batch_size):
                    
                        # Move maps to GPU
                        mps_gpu = self.mps[c:d]

                        # Apply maps
                        Sx = mps_gpu[:, None, ...] * alphas_gpu[None, ...]

                        # Apply b0 phase
                        if nseg > 1:
                            BSx = Sx[None, ...] * phase_mps_gpu[:, None, None, ...]
                        else:
                            BSx = Sx[None, ...]

                        # Zero pad image
                        BSx_os = padder.forward(BSx)

                        # FFT
                        FBSx_os = sp_fft(BSx_os, dim=tuple(range(-dim, 0)))

                        # Batch over subspace again
                        for a2, b2 in batch_iterator(nsub, self.sub_batch_size):

                            # Apply toeplitz kernels
                            if save_memory:
                                kerns = self.toep_kerns[t:u, None, a2:b2, a:b].to(self.torch_dev)
                            else:
                                kerns = toep_kerns[:, None, a2:b2, :]
                            MFBSx_os = torch.sum(kerns * FBSx_os[:, :, None, ...], dim=3)
                            
                            # Inverse FFT
                            FMFBSx_os = sp_ifft(MFBSx_os, dim=tuple(range(-dim, 0)))

                            # Crop
                            FMFBSx = padder.adjoint(FMFBSx_os)

                            # Apply adjoint maps
                            SFMFBSx = torch.sum(FMFBSx * mps_gpu[None, :, None, ...].conj(), dim=1)

                            # Apply adjoint b0 phase
                            if nseg > 1:
                                BSFMFBSx = torch.sum(SFMFBSx * phase_mps_gpu[:, None, ...].conj(), dim=0)
                            else:
                                BSFMFBSx = SFMFBSx[0]

                            # Update output
                            alphas_hat[a2:b2] += BSFMFBSx
        
        return alphas_hat
        