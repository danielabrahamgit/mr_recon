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
from mr_recon.fourier import sigpy_nufft

# Set seed
np.random.seed(0)
torch.manual_seed(0)

# Params
im_size = (220, 220)
ninter = 16
ncoil = 32
R = 4
lamda_l2 = 1e-5
max_iter = 100
device_idx = 4
torch_dev = torch.device(device_idx)

# Gen data
phantom = sp.shepp_logan(im_size)
trj = mri.spiral(fov=1,
                 N=220,
                 f_sampling=0.1,
                 R=1.0,
                 ninterleaves=ninter,
                 alpha=1.5,
                 gm=40e-3,
                 sm=100)
trj = trj.reshape((-1, ninter, 2), order='F')[:, ::round(R), :]
# trj = np.round(trj)
mps = mri.birdcage_maps((ncoil, *im_size), r=1.2)
dcf = sp.to_device(mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx)))
dcf /= dcf.max()

# Simulate with sigpy 
ksp = sp.nufft(phantom * mps, trj, oversamp=2.0, width=6)

# Add noise
sigma = 4e-3 * 2
ksp += np.random.normal(0, sigma, ksp.shape) + 1j * np.random.normal(0, sigma, ksp.shape)

# Time compression
from scipy.interpolate import interp1d
from scipy.signal import resample, resample_poly
FOV_inv = 0.9
t = np.arange(trj.shape[0])
dks = np.linalg.norm(np.diff(trj, axis=0), axis=-1)
dks_interp = interp1d(t[:-1], dks[:, 0], axis=0, fill_value='extrapolate', kind='linear')
dt = t[1] - t[0]
t_new = np.zeros(len(t))
i = 1
t_cur = 0
while t_cur < t[-1]:
    t0 = t_new[i-1]
    dk = dks_interp(t0)
    t_new[i] = t0 + dt/dk * FOV_inv 
    t_cur = t_new[i]
    i += 1
t_new = np.minimum(t_new[:i], t[-1])
# t_new = np.arange(0, len(t), 4)   
os = 2
ksp_ft = np.moveaxis(sp.ifft(ksp, axes=[1], oshape=[ksp.shape[0], ksp.shape[1]*os, ksp.shape[2]]), -2, -1)
t_trj = t_new[:, None] - trj.shape[0] / 2
trj_new = interp1d(t, trj, axis=0, fill_value='extrapolate', kind='linear')(t_new)
dks_new = np.linalg.norm(np.diff(trj_new, axis=0), axis=-1)
# ksp_new = np.moveaxis(sp.nufft(ksp_ft, t_trj), -2, -1)
nft = sigpy_nufft((trj.shape[0],), apodize=False)
ksp_new = nft(np_to_torch(ksp_ft)[None,], np_to_torch(t_trj)[None,]).cpu().moveaxis(-2, -1).numpy()[0]
plt.figure(figsize=(14, 7))
plt.subplot(211)
plt.plot(t[:-1], dks[:, 0], label='Original')
plt.subplot(212)
plt.plot(t_new[:-1], dks_new[:, 0], label='Interpolated', color='r', alpha=0.6)
plt.figure(figsize=(14, 7))
plt.plot(t, ksp.real[0, :, 0], label='original')
plt.scatter(t_new, ksp_new.real[0, :, 0], color='r', marker='.', label='interpolated')
plt.legend()
# quit()
# t_new = np.arange(0, len(t), 4)
# trj_new = interp1d(t, trj, axis=0, fill_value='extrapolate', kind='linear')(t_new)
# ksp_new = resample_poly(ksp, 1, 4, axis=1)
dcf_new = sp.to_device(mri.pipe_menon_dcf(trj_new, im_size, device=sp.Device(device_idx)))
dcf_new /= dcf_new.max()
print(trj.shape)
print(trj_new.shape)

# Regular Recon
bparams = batching_params(coil_batch_size=ncoil)
nufft = sigpy_nufft(im_size=im_size)
A = sense_linop(im_size=im_size,
                trj=np_to_torch(trj).to(torch_dev),
                mps=np_to_torch(mps).to(torch_dev),
                dcf=np_to_torch(dcf).to(torch_dev),
                bparams=bparams,
                nufft=nufft,
                use_toeplitz=False,)
img = CG_SENSE_recon(A=A, 
                     ksp=np_to_torch(ksp).to(torch_dev), 
                     lamda_l2=lamda_l2,
                     max_eigen=1.0,
                     max_iter=max_iter).cpu().numpy()

# Time compressed recon
A_tc = sense_linop(im_size=im_size,
                   trj=np_to_torch(trj_new).to(torch_dev),
                   mps=np_to_torch(mps).to(torch_dev),
                   dcf=np_to_torch(dcf_new).to(torch_dev),
                   bparams=bparams,
                   nufft=nufft,
                   use_toeplitz=False,)
img_tc = CG_SENSE_recon(A=A_tc, 
                        ksp=np_to_torch(ksp_new).to(torch_dev), 
                        lamda_l2=lamda_l2,
                        max_eigen=1.0,
                        max_iter=max_iter).cpu().numpy()


# Compare
img_tc *= np.linalg.norm(img) / np.linalg.norm(img_tc)
vmax = np.abs(img).max() * 0.8
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('Standard')
plt.imshow(np.abs(img), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(222)
plt.title('Time Compressed')
plt.imshow(np.abs(img_tc), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(223)
plt.title(f'Trajectory (R = {R})')
for i in range(trj.shape[1]):
    plt.plot(trj[:, i, 0], trj[:, i, 1], color='black', alpha=0.2)
plt.plot(trj[:, 0, 0], trj[:, 0, 1], color='red')
plt.axis('off')
plt.subplot(224)
plt.title('Difference (5X)')
plt.imshow(np.abs(img - img_tc), cmap='gray', vmin=0, vmax=vmax/5)
plt.axis('off')
plt.tight_layout()
plt.show()
