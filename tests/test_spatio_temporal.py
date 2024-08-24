import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_sim.coil_maps import surface_coil_maps
from mr_sim.grad_utils import design_spiral_trj

from mr_recon.algs import density_compensation
from mr_recon.fourier import fft, ifft
from mr_recon.spatial import apply_window
from mr_recon.utils import np_to_torch, gen_grd
from mr_recon.imperfections.spatio_temporal_imperf import B0
from mr_recon.imperfections.main_field import main_field_imperfection
from mr_recon.linops import sense_linop
from mr_recon.recons import CG_SENSE_recon
from mr_recon.imperfections.imperf_decomp import svd_decomp_fast_temporal, temporal_segmentation

# Params
gmax, smax = 40e-3, 150
res, fov = 1e-3, 0.22
dt = 2e-6
R = 2
nshots = 4
dt = 2e-6
L = 3
torch_dev = torch.device(6)

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
# trj = design_epi_trj(gmax=gmax, smax=smax, res=res, fov_y=fov, nshots=nshots, dt=dt)[:-1, :, :2]
trj = design_spiral_trj(gmax=gmax, smax=smax, res=res, fov=fov, nshots=nshots, alpha=1.0, dt=dt)
trj = trj[:, ::R, :].to(torch_dev).flip(-1) * fov
dcf = density_compensation(trj, im_size)
print(f'Readout Time = {trj.shape[0]* dt * 1e3} ms')

b0_imperf = B0(b0.shape, trj.shape[:-1], readout_dim=0, b0_map=b0 * dt)
print(f'Computing decomposition ... ', end='')
b, h = temporal_segmentation(b0_imperf, L, interp_type='lstsq')
# b, h = svd_decomp_fast_temporal(b0_imperf, L)
print(f'done.')

# Simulate data
imperf_model = main_field_imperfection(b0, trj.shape[:-1], 0, dt=2e-6, L=40, method='svd', interp_type='lstsq')
A = sense_linop(im_size, trj, mps, dcf, 
                imperf_model=imperf_model)
ksp = A(img)

# Recon with imperfection model
# A.imperf_model.temporal_funcs = A.imperf_model.temporal_funcs[:L]
# A.imperf_model.spatial_funcs = A.imperf_model.spatial_funcs[:L]
# A.imperf_model.temporal_funcs = h
# A.imperf_model.spatial_funcs = b
imperf_model = main_field_imperfection(b0, trj.shape[:-1], 0, dt=2e-6, L=L, method='ts', interp_type='lstsq')
A.imperf_model = imperf_model
A.imperf_rank = L
recon = CG_SENSE_recon(A, ksp)

# Show
plt.imshow(recon.abs().cpu().T.flip(dims=(0,)), cmap='gray')
plt.show()
