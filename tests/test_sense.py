import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri
from mr_recon.recon import recon
from mr_recon.linop import subspace_linop

# Params
im_size = (220, 220)
ninter = 16
ncoil = 32
R = 4
lamda_l2 = 1e-3 * 0
device_idx = 5
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
trj = np.round(trj)
mps = mri.birdcage_maps((ncoil, *im_size), r=1.2)
dcf = sp.to_device(mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx)))
dcf /= dcf.max()

# Simulate with sigpy 
ksp = sp.nufft(phantom * mps, trj, oversamp=2, width=6)

# Recon with sigpy
img_sigpy = sp.to_device(mri.app.SenseRecon(ksp, mps, weights=dcf, coord=trj, lamda=lamda_l2, device=sp.Device(device_idx)).run())

# Recon with mr_recon
rcn = recon(device_idx)
phi = np.ones((1, 1))
A = subspace_linop(im_size=im_size,
                   trj=torch.tensor(trj, dtype=torch.float32, device=torch_dev)[..., None, :],
                   mps=torch.tensor(mps, dtype=torch.complex64, device=torch_dev),
                   phi=torch.tensor(phi, dtype=torch.complex64, device=torch_dev),
                   dcf=torch.tensor(dcf, dtype=torch.float32, device=torch_dev)[..., None],
                   use_toeplitz=True,
                   grog_grid_oversamp=1.0,
                   coil_batch_size=ncoil)
img_mr_recon = rcn.run_recon(A_linop=A,
                             ksp=ksp[..., None],
                             max_eigen=1.0,
                             max_iter=100,
                             lamda_l2=lamda_l2)[0]

img_sigpy *= np.linalg.norm(img_mr_recon) / np.linalg.norm(img_sigpy)
vmax = np.abs(img_sigpy).max() * 0.9
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.imshow(np.abs(img_sigpy), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(222)
plt.imshow(np.abs(img_mr_recon), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(223)
for i in range(trj.shape[1]):
    plt.plot(trj[:, i, 0], trj[:, i, 1], color='black', alpha=0.2)
plt.plot(trj[:, 0, 0], trj[:, 0, 1], color='red')
plt.axis('off')
plt.subplot(224)
plt.imshow(np.abs(img_mr_recon - img_sigpy), cmap='gray', vmin=0, vmax=vmax/5)
plt.axis('off')
plt.tight_layout()
plt.show()
