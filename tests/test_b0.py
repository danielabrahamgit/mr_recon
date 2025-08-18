import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.imperfections.field import b0_to_phis_alphas, alpha_segementation
from mr_recon.prox import L1Wav, LocallyLowRank, LLRHparams
from mr_recon.algs import density_compensation
from mr_recon.recons import CG_SENSE_recon, FISTA_recon
from mr_recon.linops import sense_linop, subspace_linop, batching_params
from mr_recon.fourier import sigpy_nufft, svd_nufft, fft, ifft
from mr_recon.utils import np_to_torch
from mr_recon.spatial import apply_window
from mr_sim.phantoms import shepp_logan
from mr_sim.grad_utils import design_epi_trj, design_spiral_trj

# Params
gmax = 40e-3
smax = 150
fov = 0.22
dt = 2e-6
shots = 4
res = 1e-3
R = 1
torch_dev = torch.device(1)

nufft = svd_nufft((100,))
trj = torch.rand((1000, 1), dtype=torch.float32, device=torch_dev) - 0.5
nufft.rescale_trajectory(trj)

# Load b0 map from scan
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/mrf_b0/data/'
zslc = 110
img = np_to_torch(np.load(fpath + 'img.npy')).type(torch.complex64).to(torch_dev)[2, ..., zslc]
b0 = np_to_torch(np.load(fpath + 'b0.npy')).type(torch.float32).to(torch_dev)[..., zslc]
mps = np_to_torch(np.load(fpath + 'mps.npy')).type(torch.complex64).to(torch_dev)[..., zslc]
im_size = img.shape

# Smooth out b0 a bit
crp_size = 64
crp = tuple([slice(im_size[i]//2 - crp_size//2, im_size[i]//2 + crp_size//2) for i in range(len(im_size))])
b0_ft = apply_window(fft(b0)[crp], ndim=2)
b0 = ifft(b0_ft, oshape=img.shape).real

# Design trajectory
# trj = design_epi_trj(gmax=gmax, smax=smax, res=res, fov_y=fov, nshots=shots, dt=dt)[:-1, :, :2]
trj = design_spiral_trj(gmax=gmax, smax=smax, res=res, fov=fov, nshots=shots, alpha=1.0, dt=dt)
trj = trj[:, ::R, :].to(torch_dev).flip(-1) * fov
dcf = density_compensation(trj, im_size)
print(f'Readout Time = {trj.shape[0]* dt * 1e3} ms')

# B0 model
phis, alphas = b0_to_phis_alphas(b0, dcf.shape, 0, dt)
b, h = alpha_segementation(phis, alphas, L=50, interp_type='lstsq', use_type3=False)
A_sim = sense_linop(trj, mps, dcf, spatial_funcs=b, temporal_funcs=h)
ksp = A_sim(img)

# Linop
b, h = alpha_segementation(phis, alphas, L=20, interp_type='lstsq', use_type3=False)
nufft = sigpy_nufft(im_size)
bparams = batching_params(coil_batch_size=mps.shape[0])
A = sense_linop(trj, mps, dcf, nufft, bparams=bparams, spatial_funcs=b, temporal_funcs=h, use_toeplitz=True)
A_nufft = sense_linop(trj, mps, dcf, nufft, bparams=bparams)

# Recons
img_artifact = CG_SENSE_recon(A_nufft, ksp, max_iter=15,
                    #  max_eigen=1,
                     lamda_l2=1e-2,)
img_recon = CG_SENSE_recon(A, ksp, max_iter=15,
                    #  max_eigen=1,
                     lamda_l2=1e-2,)

# Image Plot
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(img.abs().cpu().T, cmap='gray', origin='lower')
plt.axis('off')
plt.subplot(222)
plt.imshow(img_artifact.abs().cpu().T, cmap='gray', origin='lower')
plt.axis('off')
plt.subplot(223)
plt.imshow(img_recon.abs().cpu().T, cmap='gray', origin='lower')
plt.axis('off')
plt.subplot(224)
plt.imshow(b0.cpu().T, cmap='jet', origin='lower')
plt.colorbar()
plt.axis('off')
plt.tight_layout()

# # Trajectory Plot
# plt.figure(figsize=(7,7))
# plt.title(f'Trajectory Readout Time = {trj.shape[0]* dt * 1e3:.1f} ms')
# for i in range(shots):
#     plt.plot(trj[:, i, 1].cpu(), trj[:, i, 0].cpu(), color='black', alpha=0.2)
# plt.plot(trj[:, 0, 1].cpu(), trj[:, 0, 0].cpu(), color='red')
# plt.axis('off')
# plt.tight_layout()
plt.show()