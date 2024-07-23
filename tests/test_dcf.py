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
from mr_recon.fourier import sigpy_nufft, gridded_nufft, torchkb_nufft
from mr_recon.algs import density_compensation
from mr_sim.phantom import shepp_logan

# Params
im_size = (220, 220)
ncoil = 32
R = 6
ninter = R
lamda_l2 = 1e-5 * 0
max_iter = 100
device_idx = 5
try:
    torch_dev = torch.device(device_idx)
except:
    torch_dev = torch.device('cpu')

# Gen data
phantom = shepp_logan(torch_dev).img(im_size)
trj = mri.spiral(fov=1,
                 N=220,
                 f_sampling=1.0,
                 R=1.0,
                 ninterleaves=ninter,
                 alpha=1.5,
                 gm=40e-3,
                 sm=100)
trj = trj.reshape((-1, ninter, 2), order='F')[:, ::round(R), :]
trj = np_to_torch(trj).type(torch.float32).to(torch_dev)
mps = mri.birdcage_maps((ncoil, *im_size), r=1.2)
mps = np_to_torch(mps).type(torch.complex64).to(torch_dev)
dcf = density_compensation(trj, im_size, 7, method='cg_img')

# Simulate with sigpy nufft
spnufft = sigpy_nufft(im_size, device_idx, oversamp=2.0, width=6) 
ksp = spnufft.forward((phantom * mps)[None,], trj[None,])[0]

# Recon with mr_recon
bparams = batching_params(coil_batch_size=ncoil)
nufft = sigpy_nufft(im_size=im_size, device_idx=device_idx)
# nufft = torchkb_nufft(im_size=im_size, device_idx=device_idx)
# nufft = gridded_nufft(im_size=im_size, device_idx=device_idx)
A = sense_linop(im_size=im_size,
                trj=trj,
                mps=mps,
                dcf=dcf,
                bparams=bparams,
                nufft=nufft,
                use_toeplitz=False,)
img = CG_SENSE_recon(A=A, 
                     ksp=ksp, 
                     lamda_l2=lamda_l2,
                     max_eigen=1.0,
                     max_iter=max_iter).cpu().T

# Compare
plt.figure(figsize=(14, 7))
plt.imshow(img.abs(), cmap='gray')
plt.axis('off')

plt.figure(figsize=(14, 7))
trj = trj.cpu()
plt.title(f'Trajectory (R = {R})')
for i in range(trj.shape[1]):
    plt.plot(trj[:, i, 0], trj[:, i, 1], color='black', alpha=0.2)
plt.plot(trj[:, 0, 0], trj[:, 0, 1], color='red')

plt.figure(figsize=(14, 7))
plt.plot(dcf.cpu()[:, 0])
plt.show()
