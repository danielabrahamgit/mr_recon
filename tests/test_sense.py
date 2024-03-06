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
from mr_recon.fourier import sigpy_nufft, gridded_nufft

# Params
im_size = (220, 220)
ninter = 16
ncoil = 32
R = 4
lamda_l2 = 1e-3 * 0
max_iter = 100
device_idx = 4
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
# trj = np.round(trj)
mps = mri.birdcage_maps((ncoil, *im_size), r=1.2)
dcf = sp.to_device(mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx)))
dcf /= dcf.max()

# Simulate with sigpy 
ksp = sp.nufft(phantom * mps, trj, oversamp=2.0, width=6)

# Recon with sigpy
img_sigpy = sp.to_device(mri.app.SenseRecon(ksp, mps, 
                                            weights=dcf, 
                                            coord=trj, 
                                            lamda=lamda_l2, 
                                            device=sp.Device(device_idx), 
                                            max_iter=max_iter).run())

# Recon with mr_recon
bparams = batching_params(coil_batch_size=ncoil)
# nufft = gridded_nufft(im_size=im_size, device_idx=device_idx)
A = sense_linop(im_size=im_size,
                trj=np_to_torch(trj).to(torch_dev),
                mps=np_to_torch(mps).to(torch_dev),
                dcf=np_to_torch(dcf).to(torch_dev),
                bparams=bparams,
                # nufft=nufft,
                use_toeplitz=True,)
img_mr_recon = CG_SENSE_recon(A=A, 
                              ksp=np_to_torch(ksp).to(torch_dev), 
                              lamda_l2=lamda_l2,
                              max_eigen=1.0,
                              max_iter=max_iter).cpu().numpy()
# img_mr_recon = min_norm_recon(A=A, 
#                               ksp=np_to_torch(ksp).to(torch_dev), 
#                               max_eigen=1.0,
#                               max_iter=max_iter).cpu().numpy()

# Compare
img_sigpy *= np.linalg.norm(img_mr_recon) / np.linalg.norm(img_sigpy)
vmax = np.abs(img_sigpy).max() * 0.9
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('SigPy')
plt.imshow(np.abs(img_sigpy), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(222)
plt.title('mr_recon')
plt.imshow(np.abs(img_mr_recon), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(223)
plt.title('Trajectory')
for i in range(trj.shape[1]):
    plt.plot(trj[:, i, 0], trj[:, i, 1], color='black', alpha=0.2)
plt.plot(trj[:, 0, 0], trj[:, 0, 1], color='red')
plt.axis('off')
plt.subplot(224)
plt.title('Difference (5X)')
plt.imshow(np.abs(img_mr_recon - img_sigpy), cmap='gray', vmin=0, vmax=vmax/5)
plt.axis('off')
plt.tight_layout()
plt.show()
