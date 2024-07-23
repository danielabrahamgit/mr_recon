import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri
from mr_recon.linops import sense_linop, batching_params
from mr_recon.utils import np_to_torch, gen_grd
from mr_recon.multi_coil.gfactor import spatial_std_svd, spatial_std_monte_carlo
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import sigpy_nufft, gridded_nufft

# Params
ncoil = 12
R = 6
im_size = (R*(220//R), R*(220//R))
lamda_l2 = 1e-5
max_iter = 100
sigma = 1e-3
max_eigen = 1.0
device_idx = 5
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

# Move to torch
msk = np_to_torch(phantom).to(torch_dev).abs() > 0
ksp = np_to_torch(ksp).to(torch_dev).type(torch.complex64)
trj = np_to_torch(trj).to(torch_dev).type(torch.float32)
mps = np_to_torch(mps).to(torch_dev).type(torch.complex64) * msk
ksp_noisy = ksp + sigma * torch.randn_like(ksp)

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
                     max_eigen=max_eigen,
                     max_iter=max_iter).cpu()

# Recon with undersampling
trj_us = trj[:, ::R, :]
ksp_us = ksp[..., ::R]
ksp_us_noisy = ksp_noisy[..., ::R]
bparams = batching_params(coil_batch_size=ncoil)
nufft = gridded_nufft(im_size=im_size, device_idx=device_idx)
A_us = sense_linop(im_size=im_size,
                   trj=trj_us,
                   mps=mps,
                   bparams=bparams,
                   nufft=nufft,
                   use_toeplitz=False,)
recon_alg = lambda inp: CG_SENSE_recon(A=A_us, ksp=inp, lamda_l2=lamda_l2,max_eigen=max_eigen,max_iter=max_iter, verbose=False)
img_us = recon_alg(ksp_us).cpu()
img_us_noisy = recon_alg(ksp_us_noisy).cpu()

# Compute g-factor
std_svd = spatial_std_svd(AHA=A_us.normal, im_size=im_size, lambda_l2=lamda_l2, device=torch_dev)
std_monte = spatial_std_monte_carlo(recon_alg=recon_alg, im_size=im_size, data_size=ksp_us.shape, device=torch_dev, n_monte_carlo=1000, sigma=sigma)
std_svd *= msk

plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.imshow(std_svd.cpu(), cmap='jet')#, vmin=0, vmax=1)
plt.title('SVD')
plt.colorbar()
plt.subplot(122)
plt.imshow(std_monte.cpu(), cmap='jet')
plt.title('Monte Carlo')
plt.colorbar()

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
plt.imshow(np.abs(img_us - img_us_noisy), cmap='jet')
plt.axis('off')
plt.subplot(224)
plt.title('Error')
plt.imshow(np.abs(img - img_us), cmap='gray')
plt.axis('off')
plt.show()