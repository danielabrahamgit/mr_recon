import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri
from mr_recon.linops import sense_linop, batching_params
from mr_recon.utils import np_to_torch, gen_grd
from mr_recon.gfactor import calc_variance_mc, calc_variance_analytic
from mr_recon.recons import CG_SENSE_recon, min_norm_recon
from mr_recon.fourier import sigpy_nufft, gridded_nufft

# Params
im_size = (200, 200)
ncoil = 12
R = 3
lamda_l2 = 1e-3 * 0
sigma = 4e-3
max_iter = 100
device_idx = -1
try:
    torch_dev = torch.device(device_idx)
except:
    torch_dev = torch.device('cpu')

# Gen data
phantom = sp.shepp_logan(im_size)
trj = gen_grd(im_size, fovs=im_size).numpy()
mps = mri.birdcage_maps((ncoil, *im_size))

# Simulate with sigpy 
ksp = sp.nufft(phantom * mps, trj, oversamp=2.0, width=6)
ksp += sigma * (np.random.randn(*ksp.shape) + 1j * np.random.randn(*ksp.shape)) / np.sqrt(R)

# Move to torch
msk = np_to_torch(phantom).to(torch_dev).abs() > 0
ksp = np_to_torch(ksp).to(torch_dev).type(torch.complex64)
trj = np_to_torch(trj).to(torch_dev).type(torch.float32)
mps = np_to_torch(mps).to(torch_dev).type(torch.complex64) * msk

# Recon with no undersampling
bparams = batching_params(coil_batch_size=ncoil)
nufft = gridded_nufft(im_size=im_size, device_idx=device_idx)
A = sense_linop(im_size=im_size,
                trj=trj,
                mps=mps,
                bparams=bparams,
                nufft=nufft,
                use_toeplitz=False,)
img = CG_SENSE_recon(A=A, 
                     ksp=ksp, 
                     lamda_l2=lamda_l2,
                     max_eigen=1.0,
                     max_iter=max_iter).cpu().numpy()
# var = calc_variance_mc(A.normal, A.adjoint, ksp, sigma_noise=sigma).cpu().numpy()
var = calc_variance_analytic(A.normal, im_size, torch_dev, sigma_noise=sigma, msk=msk).cpu().numpy()

# Recon with undersampling
trj_us = trj[:, ::R, :]
ksp_us = ksp[..., ::R]
bparams = batching_params(coil_batch_size=ncoil)
nufft = gridded_nufft(im_size=im_size, device_idx=device_idx)
A_us = sense_linop(im_size=im_size,
                   trj=trj_us,
                   mps=mps,
                   bparams=bparams,
                   nufft=nufft,
                   use_toeplitz=False,)
img_us = CG_SENSE_recon(A=A_us, 
                        ksp=ksp_us, 
                        lamda_l2=lamda_l2,
                        max_eigen=1.0,
                        max_iter=max_iter).cpu().numpy()
# var_us = calc_variance_mc(A_us.normal, A_us.adjoint, ksp_us, sigma_noise=sigma).cpu().numpy()
var_us = calc_variance_analytic(A_us.normal, im_size, torch_dev, sigma_noise=sigma, msk=msk).cpu().numpy()

# Compute g-factor
gfact = np.sqrt(var_us / var / R)

# Compare
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('Fully Sampled')
plt.imshow(np.abs(img), cmap='gray')
plt.axis('off')
plt.subplot(222)
plt.title(f'R = {R}')
plt.imshow(np.abs(img_us), cmap='gray')
plt.axis('off')
plt.subplot(223)
plt.title('G-Factor')
plt.imshow(gfact, cmap='jet')
plt.axis('off')
plt.subplot(224)
plt.title('Error')
plt.imshow(np.abs(img - img_us), cmap='gray')
plt.axis('off')
# plt.title('Trajectory')
# trj = trj.cpu().numpy()
# for i in range(trj.shape[1]):
#     plt.plot(trj[:, i, 0], trj[:, i, 1], color='black', alpha=0.2)
# plt.plot(trj[:, 0, 0], trj[:, 0, 1], color='red')
# plt.axis('off')
plt.show()