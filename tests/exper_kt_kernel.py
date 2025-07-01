from mr_recon.spatial import spatial_resize, spatial_interp
import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_sim.field_sim import sim_b0
from mr_sim.coil_maps import surface_coil_maps

from mr_recon.utils import batch_iterator, gen_grd, normalize
from mr_recon.recons import CG_SENSE_recon
from mr_recon.algs import density_compensation
from mr_recon.fourier import sigpy_nufft, NUFFT, svd_nufft
from mr_recon import dtypes
from mr_recon.linops import batching_params, experimental_sense, linop
from mr_recon.imperfections.imperf_decomp import temporal_segmentation
from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase

from typing import Optional

# Set seeds
torch.manual_seed(0)

# Params
C = 8
im_size = (276, 276)
dt = 2e-6
torch_dev = torch.device(4)

# Create fake data
pth = '/local_mount/space/mayday/data/users/abrahamd/hofft/coco_spiral/best_slice/'
trj = torch.load(pth+'trj.pt', map_location=torch_dev, weights_only=True)
# dcf = torch.load(pth+'dcf.pt', map_location=torch_dev, weights_only=True)
dcf = density_compensation(trj, im_size)
img = shepp_logan(torch_dev).img(im_size)
mps = surface_coil_maps(C, im_size, img, torch_dev=torch_dev)
b0 = sim_b0(im_size, b0_max=60, method='brown').to(torch_dev) * (img.abs() > 0)

# Simulate kspace
bparams = batching_params(C)
ts = torch.arange(trj.shape[0], device=torch_dev) * dt
ts = torch.repeat_interleave(ts[:, None], trj.shape[1], dim=-1)
hop = high_order_phase(b0[None,], ts[None,])
b, h = temporal_segmentation(hop, L=100, interp_type='lstsq')
Asim = experimental_sense(trj, mps, 
                          spatial_funcs=b, 
                          temporal_funcs=h,
                          dcf=None,
                          bparams=bparams)
ksp = Asim(img)

# Write custom linop
class kt_3d(linop):
    
    def __init__(self,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 b0: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        trj : torch.tensor <float>
            The k-space trajectory with shape (*trj_size, 2). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
                we also assume that trj_size[0] is the readout dimension (for b0 phase)
        mps : torch.tensor <complex>
            the coil sensitivity with shape (L, *im_size)
        b0 : torch.tensor <float>
            the b0 field with shape (*im_size) units of 1/samples
        dcf : torch.tensor <float>
            the density comp. functon with shape (*trj_size)
        nufft : NUFFT
            the nufft object, defaults to sigpy_nufft
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains various batch sizes
        """
        im_size = mps.shape[1:]
        trj_size = trj.shape[:-1]
        C = mps.shape[0]
        super().__init__(im_size, (C, *trj_size))

        # Consts
        torch_dev = trj.device
        assert mps.device == torch_dev
        assert b0.device == torch_dev

        # Default params
        W = 6
        os = 1.25
        if nufft is None:
            nufft = sigpy_nufft(im_size, oversamp=os, width=W)
        if dcf is None:
            dcf = torch.ones(trj_size, dtype=dtypes.real_dtype, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Rescale and change types
        trj = nufft.rescale_trajectory(trj).type(dtypes.real_dtype)
        dcf = dcf.type(dtypes.real_dtype)
        
        # Compute 3D trajectory 
        b0_scale = b0.abs().max() * 2
        b0 = b0 / b0_scale
        ts = torch.arange(trj.shape[0], device=torch_dev) * b0_scale
        tup = (slice(None),) + (None,) * (len(trj_size) - 1)
        ts = torch.tile(ts[tup], (1, *trj_size[1:]))
        self.trj_3d = torch.cat([trj, ts[..., None]], dim=-1) # *trj_size, 3
        
        T_size = 2 * (ts.max().ceil().int().item() + W)
        self.nufft_3d = sigpy_nufft((*im_size, T_size), width=W, oversamp=os)
        zeros = torch.zeros((*im_size, T_size), dtype=dtypes.complex_dtype, device=torch_dev)
        self.kgrid = self.nufft_3d.forward_FT_only(zeros)
        self.tsegs = gen_grd((self.kgrid.shape[-1],),(self.kgrid.shape[-1],))[:, 0].to(torch_dev) / self.nufft_3d.oversamp
        # self.tmask = torch.ones_like(self.tsegs)
        self.tmask = torch.logical_and(self.tsegs >= (ts.min() - W/2).floor(),
                                       self.tsegs <= (ts.max() + W/2).ceil()).float()
        print(self.tmask.sum())
        
        
        # Compute toeplitz kernels
        if use_toeplitz:
            raise NotImplementedError("Toeplitz kernels not implemented for this linop")

        # Save
        self.im_size = im_size
        self.trj_size = trj_size
        self.trj = trj
        self.dcf = dcf
        self.mps = mps
        self.b0 = b0
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
        C = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size

        # Result array
        ksp = torch.zeros((C, *self.trj_size), dtype=dtypes.complex_dtype, device=self.torch_dev)

        # Batch over coils 
        for c1, c2 in batch_iterator(C, coil_batch_size): 
            
            # Apply coil maps
            Sx = self.mps[c1:c2] * img # C *im_size
            
            # Apply b0 time segments & apodize
            BSx = torch.exp(-2j * torch.pi * self.b0[None,] * self.tsegs[:, None, None]) * Sx[:, None] # C T *im_size
            BSx = BSx * self.nufft_3d._apodization_func(self.b0 / self.nufft_3d.oversamp)
            
            # FT
            FBSx = self.nufft.forward_FT_only(BSx).moveaxis(1, -1) * self.tmask # C *im_size_os T
            PFBSx = self.nufft_3d.forward_interp_only(FBSx[None,], self.trj_3d[None,])[0] # C *trj_size
            
            # Append to k-space
            ksp[c1:c2] += PFBSx

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
        C = self.mps.shape[0]
        coil_batch_size = self.bparams.coil_batch_size

        # Result image
        img = torch.zeros(self.im_size, dtype=dtypes.complex_dtype, device=self.torch_dev)
        
        # Apply DCF
        Wy = (ksp * self.dcf) # C *trj_size
            
        # Batch over coils 
        for c1, c2 in batch_iterator(C, coil_batch_size): 
            
            # iFT
            PWy = self.nufft_3d.adjoint_grid_only(Wy[None,c1:c2], self.trj_3d[None,])[0] * self.tmask # C *trj_size T
            FPWy = self.nufft.adjoint_iFT_only(PWy.moveaxis(-1, 1)) # C T *im_size_os
            
            # Time segments, coil map, apodization
            SBFPWy = FPWy * mps[c1:c2, None,].conj() * torch.exp(2j * torch.pi * self.b0[None,] * self.tsegs[:, None, None]) # C T *im_size_os
            SBFPWy = SBFPWy * self.nufft_3d._apodization_func(self.b0 / self.nufft_3d.oversamp)

            # Append to image
            img += SBFPWy.sum(dim=[0,1])

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


# Reconstruct
nufft = svd_nufft(im_size, grid_oversamp=1, n_svd=20)
evecs, evals = nufft.compute_svd_funcs(trj)

# A = kt_3d(trj, mps, b0 * dt, dcf, bparams=bparams)
# b, h = temporal_segmentation(hop, L=13, interp_type='lstsq')
# A = experimental_sense(trj, mps, 
#                        nufft=sigpy_nufft(im_size, width=6),
#                        spatial_funcs=b,
#                        temporal_funcs=h,
#                        dcf=dcf,
#                        bparams=bparams)
img_est = CG_SENSE_recon(A, ksp, max_iter=15).cpu()
img = img.cpu()
img_est = normalize(img_est, img)

# Plot both
vmin = 0
vmax = img.abs().median() + 3 * img.abs().std()
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(img_est.abs().cpu().rot90(), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('Reconstructed Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img.abs().cpu().rot90(), cmap='gray',  vmin=vmin, vmax=vmax)
plt.title('Ground Truth Image')
plt.axis('off')
plt.show()



