import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import firwin
from scipy.ndimage import convolve1d
from einops import einsum, rearrange, reduce

from mr_sim.phantoms import shepp_logan
from mr_sim.coil_maps import surface_coil_maps
from mr_sim.grad_utils import design_spiral_trj

from mr_recon.algs import density_compensation
from mr_recon.fourier import fft, ifft
from mr_recon.spatial import apply_window, spatial_resize
from mr_recon.utils import np_to_torch, gen_grd, normalize
from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase
from mr_recon.imperfections.eddy import eddy_imperfection, bases
from mr_recon.imperfections.exponential import exponential_imperfection
from mr_recon.linops import sense_linop, batching_params, experimental_sense
from mr_recon.recons import CG_SENSE_recon
from mr_recon.imperfections.imperf_decomp import svd_decomp_operator, temporal_segmentation, svd_decomp_matrix

from igrog.grogify import spatio_temporal_implicit_grogify, imperfection_implicit_grogify, field_grogify, gridding_grogify
from igrog.gridding import gridding_params
from igrog.training import training_params

# Fix seeds
torch.manual_seed(0)
np.random.seed(0)

# Params
gmax, smax = 40e-3, 150
res, fov = 1e-3, 0.22
dt = 1e-6
R = 2
nshots = 4
L = 1
im_size = (220, 220)
num_coils = 12
torch_dev = torch.device(6)
# torch_dev = torch.device('cpu')

gparams = gridding_params(num_inputs=1,
                          kern_width=2,
                          grid=False,
                          interp_readout=True)
tparams = training_params(epochs=30,
                          float_precision='medium',
                          show_loss=False)

# Load skope data and trajectory
diff = 3
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/dwi_spiral_phantom/data/'
trj_np = np.load(fpath + 'trj.npy')[:, 0, 0, :]
trj_np = interp1d(np.arange(trj_np.shape[0]), trj_np, axis=0, kind='cubic')(np.arange(0, trj_np.shape[0]-1, 0.5)) # upsample to 1us
trj = np_to_torch(trj_np).type(torch.float32).to(torch_dev)
for i in range(2):
    trj[..., i] *= im_size[i] / (trj[..., i].max() - trj[..., i].min())
skp_shifted = np.load(fpath + 'skp.npy')[:, :, 0, diff*10:(diff+1)*10]

# Filter skope data
skope_inds = torch.arange(4, 16, device=torch_dev)
# skope_inds = torch.tensor([4, 6, 8, 9, 11, 13, 15], device=torch_dev, dtype=torch.long)
nro = len(trj)
h = firwin(101, 10e3, fs=1/dt)
skp_shifted = convolve1d(skp_shifted, h, axis=0, mode='constant')
skp_shifted = np_to_torch(skp_shifted).to(torch_dev)
skp_delay = 125
skp = skp_shifted[skp_delay:skp_delay + nro].type(torch.float32).mean(dim=-1)[:, skope_inds]

# DCF
dcf = density_compensation(trj, im_size).type(torch.float32)
print(f'Readout Time = {trj.shape[0]* dt * 1e3} ms')

# Make phantom + mps
img = shepp_logan(torch_dev).img(im_size).T
# from sigpy.mri import birdcage_maps
# mps = np_to_torch(birdcage_maps((num_coils, *im_size), r=1.5)).type(torch.complex64).to(torch_dev)
pts = gen_grd(im_size).to(torch_dev)
pts = torch.cat([pts, pts[..., :1]*0], dim=-1)
mps = surface_coil_maps(num_coils, pts, img)

# Generate spherical harmonic bases
grd = gen_grd(im_size, (fov,)*2).to(torch_dev)
grd = torch.cat([grd, grd[..., :1]*0], dim=-1)
phis = bases(grd[..., 0], grd[..., 1], grd[..., 2])[skope_inds]

# 'Ground Truth' Eddy model
eddy_gt = eddy_imperfection(skp, im_size, (fov,)*2, skope_inds, 
                            z_ofs=0.0, rotations=(0.0, 0.0),
                            L=3000, method='ts', interp_type='zero',
                            coord_bases=grd)
bparams = batching_params(num_coils)
A_gt = sense_linop(im_size, trj, mps, dcf, imperf_model=eddy_gt, bparams=bparams)
ksp = A_gt(img)

# New eddy model
eddy = high_order_phase(im_size, trj.shape[:-1], phis, skp.T / (2 * torch.pi), 
                        num_alpha_clusters=1000,
                        img_downsample_factor=4)
b, h = temporal_segmentation(eddy, L, interp_type='lstsq')
# b, h = svd_decomp_operator(eddy, L)
# b, h = svd_decomp_matrix(eddy, L)
b = eddy.upsample(b) * 0 + 1
h = h[:, eddy.idxs] * 0 + 1
empty = exponential_imperfection(phis, skp.T, L=L, method='ts', interp_type='zero')
empty.method = 'svd'
empty.spatial_funcs = b
empty.temporal_funcs = h
A_new = sense_linop(im_size, trj, mps, dcf, imperf_model=empty, bparams=bparams)

# Old field implicit GROG
# eddy = eddy_imperfection(skp, im_size, (fov,)*2, skope_inds, 
#                          z_ofs=0.0, rotations=(0.0, 0.0),
#                          L=1, method='ts', interp_type='zero',
#                          coord_bases=grd)
# # eddy.method = 'svd'
# # eddy.spatial_funcs = b # empty.spatial_funcs * 0 + 1
# # eddy.temporal_funcs = h # empty.temporal_funcs * 0 + 1
# dct = imperfection_implicit_grogify(ksp, trj, mps, eddy, grid_params=gparams, train_params=tparams)
# A_grog = sense_linop(im_size, trj, mps, dcf, imperf_model=eddy, bparams=bparams)
# ksp_grg = dct['ksp_grd']

# # New field implicit GROG
# eddy = high_order_phase(im_size, trj.shape[:-1], phis, skp.T / (2 * torch.pi))
# dct = spatio_temporal_implicit_grogify(ksp, trj, mps, eddy, 
#                               spatial_funcs=b, temporal_funcs=h,  
#                               grid_params=gparams, train_params=tparams)
# A_grog = sense_linop(im_size, trj, mps, dcf, imperf_model=empty, bparams=bparams)
# ksp_grg = dct['ksp_grd']

# New field GROG
dct = field_grogify(ksp, phis, -skp.T / (2 * torch.pi), mps)
A_grog = sense_linop(im_size, trj, mps, dcf, bparams=bparams)
ksp_grg = dct['ksp_grd']

# Create sense linops
# A_new = experimental_sense(trj, mps, spatial_funcs=b, temporal_funcs=h, bparams=bparams)
A = sense_linop(im_size, trj, mps, dcf, bparams=bparams)
recon_gt = CG_SENSE_recon(A, ksp)
recon_new = CG_SENSE_recon(A_new, ksp)
recon_grg = CG_SENSE_recon(A_grog, ksp_grg)

# Show both
vmax = recon_gt.abs().max()
plt.figure(figsize=(14, 7))
plt.subplot(131)
plt.imshow(recon_gt.abs().cpu(), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(132)
plt.imshow(recon_new.abs().cpu(), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(133)
plt.imshow(recon_grg.abs().cpu(), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.tight_layout()
plt.show()



