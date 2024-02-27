import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri
from mr_recon.recons import recon
from mr_recon.linops import subspace_linop
from mr_recon.fourier import sigpy_nufft, torchkb_nufft, gridded_nufft
from mr_recon.utils import np_to_torch, torch_to_np
from einops import rearrange

# Params
im_size = (220, 220)
ninter = 3
ncoil = 32
R = 1
lamda_l2 = 1e-3
device_idx = 4
torch_dev = torch.device(device_idx)

# Gen data
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
# trj = np.round(trj)
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

# Sigpy method
img_batched_sp = []
for i in range(ninter):
    kspi = sp.nufft(phantoms[i], trj[i])
    imgi = sp.nufft_adjoint(kspi * dcf[i], trj[i], phantoms[i].shape)
    img_batched_sp.append(imgi)
img_batched_sp = np.array(img_batched_sp)

# To torch
phantoms = np_to_torch(phantoms).to(torch_dev)
trj = np_to_torch(trj).to(torch_dev)
dcf = np_to_torch(dcf).to(torch_dev)

# Batched method
# nfft = sigpy_nufft(im_size, device_idx=device_idx)
nfft = torchkb_nufft(im_size, device_idx=device_idx)
# nfft = gridded_nufft(im_size, device_idx=device_idx, grid_oversamp=1.0)
trj = nfft.rescale_trajectory(trj)
ksp_batched = nfft.forward(phantoms, trj)
img_batched = nfft.adjoint(ksp_batched * dcf[:, None, :], trj).cpu().numpy()

toep = nfft.calc_teoplitz_kernels(trj, dcf)
img_batched_toep = nfft.normal_toeplitz(phantoms, toep).cpu().numpy()

img_rss_sp = np.sqrt(np.sum(np.abs(img_batched_sp) ** 2, axis=1))
img_rss = np.sqrt(np.sum(np.abs(img_batched) ** 2, axis=1))
img_rss_toep = np.sqrt(np.sum(np.abs(img_batched_toep) ** 2, axis=1))
img_rss_phantom = np.sqrt(np.sum(np.abs(phantoms.cpu().numpy()) ** 2, axis=1))

plt.figure(figsize=(14,7))
plt.suptitle('Sigpy')
for i in range(3):
    vmin = img_rss_sp[i].min()
    vmax = img_rss_sp[i].max() * 0.8
    plt.subplot(1, 3, i + 1)
    plt.imshow(img_rss_sp[i], cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')

plt.figure(figsize=(14,7))
plt.suptitle('Sigpy (Mine)')
for i in range(3):
    vmin = img_rss[i].min()
    vmax = img_rss[i].max() * 0.8
    plt.subplot(1, 3, i + 1)
    plt.imshow(img_rss[i], cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')

plt.figure(figsize=(14,7))
plt.suptitle('Sigpy Toeplitz')
for i in range(3):
    vmin = img_rss_toep[i].min()
    vmax = img_rss_toep[i].max() * 0.8
    plt.subplot(1, 3, i + 1)
    plt.imshow(img_rss_toep[i], cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
plt.show()

