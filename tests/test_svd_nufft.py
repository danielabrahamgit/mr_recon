import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri
from mr_recon.linops import sense_linop, batching_params
from mr_recon.utils import np_to_torch
from mr_recon.recons import CG_SENSE_recon, min_norm_recon
from mr_recon.fourier import torchkb_nufft, gridded_nufft, sigpy_nufft
from mr_recon.imperfections.off_grid import off_grid_imperfection
from mr_recon.imperfections.main_field import main_field_imperfection

# Params
im_size = (220, 220)
ninter = 16
ncoil = 32
R = 4
lamda_l2 = 1e-3 * 0
max_iter = 100
device_idx = 6
torch_dev = torch.device(device_idx)

# Gen data
phantom = sp.shepp_logan(im_size)
trj = mri.spiral(fov=1,
                 N=220,
                 f_sampling=1.0,
                 R=1.0,
                 ninterleaves=ninter,
                 alpha=1.0,
                 gm=40e-3,
                 sm=100)
trj = trj.reshape((-1, ninter, 2), order='F')[:, ::round(R), :]
mps = mri.birdcage_maps((ncoil, *im_size), r=1.2)
dcf = sp.to_device(mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx)))
dcf /= dcf.max()

# Simulate with sigpy 
ksp = sp.nufft(phantom * mps, trj, oversamp=2.0, width=6)

# Move everything to torch
trj = np_to_torch(trj).to(torch_dev).type(torch.float32)
mps = np_to_torch(mps).to(torch_dev).type(torch.complex64)
dcf = np_to_torch(dcf).to(torch_dev).type(torch.float32)
ksp = np_to_torch(ksp).to(torch_dev).type(torch.complex64)

# Make imperfection model
trj_grd = torch.round(trj)
grid_deviations = trj - trj_grd
imperf_model = off_grid_imperfection(im_size=im_size,
                                     grid_deviations=grid_deviations,
                                     L=4,
                                     method='svd',
                                     interp_type='lstsq',
                                     verbose=True,)

# Recon with mr_recon
bparams = batching_params(coil_batch_size=ncoil, field_batch_size=2)
nufft = gridded_nufft(im_size=im_size, device_idx=device_idx)
# nufft = torchkb_nufft(im_size=im_size, device_idx=device_idx)
# nufft = sigpy_nufft(im_size=im_size, device_idx=device_idx)
A = sense_linop(im_size=im_size,
                trj=trj_grd,
                mps=mps,
                dcf=dcf,
                bparams=bparams,
                nufft=nufft,
                imperf_model=imperf_model,
                use_toeplitz=False,)
img_mr_recon = CG_SENSE_recon(A=A, 
                              ksp=np_to_torch(ksp).to(torch_dev), 
                              lamda_l2=lamda_l2,
                            #   max_eigen=1.0,
                              max_iter=max_iter).cpu().numpy()
# Show results
plt.figure(figsize=(14, 7))
plt.title('mr_recon')
img = np.abs(img_mr_recon)
vmax = np.median(img) + 3 * np.std(img)
plt.imshow(img, cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.tight_layout()
plt.show()
