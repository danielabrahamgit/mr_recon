import torch
import numpy as np
import sigpy.mri as mri

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from scipy.special import lambertw
from mr_recon.utils import np_to_torch, torch_to_np, gen_grd
from mr_sim.phantoms import shepp_logan
from math import ceil
from mr_recon.fourier import chebyshev_nufft, sigpy_nufft, gridded_nufft, svd_nufft, ifft
from einops import einsum


# SEt seeds
torch.manual_seed(0)
np.random.seed(0)

im_size = (220, 220)
L = 5
gamma = 0.5

eps = 1e-4
K = max(3, ceil(5 * gamma * np.exp(lambertw(np.log(140 / eps) / (5 * gamma)).real)))
L = K
print(f'Rank = {L}')

# Make trajectory
R = 1
trj = mri.spiral(fov=1,
                 N=220,
                 f_sampling=1.0,
                 R=1.0,
                 ninterleaves=R,
                 alpha=1.0,
                 gm=40e-3,
                 sm=100)
trj = trj.reshape((-1, R, 2), order='F')[:, 0, :]
trj = np_to_torch(trj)
r = gen_grd(im_size)

# input image
img = shepp_logan().img(im_size).T

# Apply with regular nufft
nufft_sp = sigpy_nufft(im_size)
trj_rs_sp = nufft_sp.rescale_trajectory(trj)
ksp_sp = nufft_sp(img[None,], trj_rs_sp[None,])[0] 

# Apply with lowrank nufft
# nufft = chebyshev_nufft(im_size, L)
nufft = svd_nufft(im_size, 1.0, L)
trj_rs = nufft.rescale_trajectory(trj)
ksp_cheby = nufft(img[None,], trj_rs[None,])[0] 

# Apply adjoint
img_sp = nufft_sp.adjoint(ksp_sp[None,], trj_rs_sp[None,])[0]
img_cheby = nufft.adjoint(ksp_cheby[None,], trj_rs[None,])[0]
# img_sp = ifft(ksp_sp)
# img_cheby = ifft(ksp_cheby)

# Plot images
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.imshow(img_sp.abs(), cmap='gray')
plt.title('SigPy')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_cheby.abs(), cmap='gray')
plt.title('Cheby')
plt.axis('off')

# Plot both
plt.figure(figsize=(14,7))
plt.plot(ksp_cheby.real, label='Cheby')
plt.plot(ksp_sp.real, label='GT', linestyle='--')
plt.legend()
plt.figure(figsize=(14,7))
plt.plot(ksp_cheby.imag, label='Cheby')
plt.plot(ksp_sp.imag, label='GT', linestyle='--')
plt.legend()
plt.show()
