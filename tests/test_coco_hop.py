import time
import torch
import sigpy as sp
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase
from mr_recon.imperfections.imperf_decomp import svd_decomp_operator, temporal_segmentation, svd_decomp_matrix
from mr_recon.linops import batching_params, experimental_sense
from mr_recon.recons import CG_SENSE_recon
from mr_recon.imperfections.coco import coco_imperfection

# Set seeds
torch.manual_seed(0)
np.random.seed(0)

# load data
torch_dev = torch.device(4)
# torch_dev = torch.device('cpu')
R = 3
dt = 2e-6
fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/coco_spiral/best_slice/'
kwargs = {'map_location': torch_dev, 'weights_only': True}
b0 = torch.load(f'{fpath}b0.pt', **kwargs).type(torch.float32)
mps = torch.load(f'{fpath}mps.pt', **kwargs).type(torch.complex64)
ksp = torch.load(f'{fpath}ksp.pt', **kwargs)[:, :, ::R].type(torch.complex64)
trj = torch.load(f'{fpath}trj.pt', **kwargs)[:, ::R].type(torch.float32)
dcf = torch.load(f'{fpath}dcf.pt', **kwargs)[:, ::R].type(torch.float32)
img_gt = torch.load(f'{fpath}img_gt.pt', **kwargs).type(torch.complex64)
C = ksp.shape[0]
im_size = img_gt.shape
msk = 1 * (mps.abs().sum(dim=0) > 0).to(torch_dev)
# msk = None

# Get phis and alphas 
coco_imperf = coco_imperfection(trj, im_size, fov=(0.22,)*2, dt=dt, B0=3.0, z_ofs=0.0, rotations=(0,)*2,
                                    L=1,
                                    method='ts',
                                    interp_type='zero',
                                    verbose=True)
phis = coco_imperf.phis[:-2] # ignore last term
alphas = coco_imperf.alphas[:-2] # ignore last term

print(f'ksp.shape: {ksp.shape}')
print(f'mps.shape: {mps.shape}')
print(f'trj.shape: {trj.shape}')
print(f'dcf.shape: {dcf.shape}')
print(f'b0.shape: {b0.shape}')
print(f'phis.shape: {phis.shape}')
print(f'alphas.shape: {alphas.shape}')

# Params
max_iter = 10
lamda_l2 = 1e-5
max_eigen = 1.0
verbose = False

# Scale b0
b0_scaled = b0 * trj.shape[0] * dt
ts_scaled = (torch.arange(trj.shape[0], device=torch_dev, dtype=torch.float32) + (6.5e-3 / dt)) / trj.shape[0]
phis = torch.cat([phis, b0_scaled[None,]], dim=0)
alphas = torch.cat([alphas, ts_scaled[None, :, None]], dim=0)
# phis = phis[-1:]
# alphas = alphas[-1:]

# plt.imshow(b0_scaled.cpu().T)
# plt.show()
# quit()

# u, s, vh = torch.linalg.svd(alphas.reshape((alphas.shape[0], -1)), full_matrices=False)
# # comp = u[:, :3].T
# comp = torch.eye(alphas.shape[0], device=torch_dev)
# phis = einsum(comp, phis, 'Bo Bi, Bi ... -> Bo ...')
# alphas = einsum(comp, alphas, 'Bo Bi, Bi ... -> Bo ...')

# Compute bases using HOP model
from mr_recon.spatial import spatial_resize
L = 10
# phis = spatial_resize(phis, (100,)*2, method='bicubic')
hop = high_order_phase(phis, alphas, use_KB=True)#, spatial_batch_size=im_size[0] * 20)
# b, h = svd_decomp_matrix(hop, L=L)
b, h = svd_decomp_operator(hop, L=L, fast_axis='spatial')
# b, h = temporal_segmentation(hop, L=L, interp_type='lstsq')
b = spatial_resize(b, im_size, method='bicubic')

# plt.figure(figsize=(14,7))
# for l in range(L):
#     plt.subplot(2,5,l+1)
#     plt.title(f'l = {l+1}')
#     plt.imshow(b[l].cpu().angle(), cmap='jet')#, vmin=0, vmax=2.0)
#     plt.axis('off')
# plt.subplots_adjust(wspace=0.0, hspace=0.0)
# plt.tight_layout()
# plt.show()
# quit()
    

# plt.figure(figsize=(14,7))
# h /= h.abs().max().item() * 1.5
# for l in range(L):
#     plt.axhline(l+1, color='black', alpha=0.3)
#     # plt.plot(h[l].cpu().abs() + l + 1)
#     p = plt.plot(h[l].cpu().real + l+1)
#     plt.plot(h[l].cpu().imag + l+1, ls='--', color=p[0].get_color())
# plt.yticks(range(1, L+1), [f'l = {l}' for l in range(1, L+1)])
# plt.show()
# quit()

# Apply bases to A and reconstruct
bparams = batching_params(C * 0 + 1)
A = experimental_sense(trj, mps, dcf, 
                       spatial_funcs=b, 
                       temporal_funcs=h,
                       bparams=bparams,)
img_recon = CG_SENSE_recon(A, ksp, max_iter, lamda_l2, max_eigen, verbose).cpu().T

plt.figure(figsize=(7,7))
plt.imshow(img_recon.abs(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
