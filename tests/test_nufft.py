import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri
from mr_recon.recon import recon
from mr_recon.linop import subspace_linop
from mr_recon.nufft import sigpy_nufft, torchkb_nufft, gridded_nufft
from einops import rearrange

# Params
im_size = (220, 220)
ninter = 3
ncoil = 32
R = 1
lamda_l2 = 1e-3
device_idx = 5
torch_dev = torch.device(device_idx)

# Gen 5
phantom = sp.shepp_logan(im_size)
trj = mri.spiral(fov=1,
                 N=220,
                 f_sampling=1.0,
                 R=0.3,
                 ninterleaves=ninter,
                 alpha=1.0,
                 gm=40e-3,
                 sm=100)
trj = trj.reshape((-1, ninter, 2), order='F')[:, ::round(R), :]
trj = rearrange(trj, 'nro nint d -> nint nro d')
trj = np.round(trj)
mps = mri.birdcage_maps((ncoil, *im_size), r=1.2)
dcf = sp.to_device(mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx)))
dcf /= dcf.max()

# Gen image for each interleave
phantoms = []
for i in range(ninter):
    new_phantom = phantom ** (i + 1)
    new_phantom /= np.abs(new_phantom).max()
    phantoms.append(new_phantom * mps)
phantoms = np.array(phantoms)

ksp_for = []
for i in range(ninter):
    ksp_for.append(sp.nufft(phantoms[i], trj[i]))
ksp_for = np.array(ksp_for)

# Batched method
nfft = sigpy_nufft(im_size)
# nfft = torchkb_nufft(im_size)
# nfft = gridded_nufft(im_size, grid_oversamp=1.0)
ksp_batched = nfft.forward(phantoms, trj).numpy()
img_batched = nfft.adjoint(ksp_batched * dcf[:, None, :], trj).numpy()

img_rss = np.sqrt(np.sum(np.abs(img_batched) ** 2, axis=1))

for i in range(3):
    plt.figure()
    plt.imshow(img_rss[i])
plt.show()

