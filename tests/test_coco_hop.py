import time
import torch
import sigpy as sp
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from einops import einsum, rearrange

from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase
from mr_recon.imperfections.imperf_decomp import svd_decomp_operator, temporal_segmentation, svd_decomp_matrix, decomp_with_grappa
from mr_recon.linops import batching_params, experimental_sense, experimental_sense_coil
from mr_recon.recons import CG_SENSE_recon
from mr_recon.imperfections.coco import coco_imperfection
from mr_recon.calib import synth_cal
from mr_recon.fourier import ifft

from igrog.grogify import spatio_temporal_implicit_grogify
from igrog.gridding import gridding_params
from igrog.training import training_params

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
img_gt = torch.load(f'{fpath}img_gt.pt', **kwargs).type(torch.complex64)
ksp_fs = torch.load(f'{fpath}ksp.pt', **kwargs).type(torch.complex64)
trj_fs = torch.load(f'{fpath}trj.pt', **kwargs).type(torch.float32)
dcf_fs = torch.load(f'{fpath}dcf.pt', **kwargs).type(torch.float32)
ksp = ksp_fs[..., ::R]
trj = trj_fs[:, ::R]
dcf = dcf_fs[:, ::R]
C = ksp.shape[0]
im_size = img_gt.shape
msk = 1 * (mps.abs().sum(dim=0) > 0).to(torch_dev)
# msk = None

# Calib data
ksp_cal = synth_cal(ksp_fs, (32, 32), trj_fs, dcf_fs, num_iter=1)
img_cal = ifft(ksp_cal, dim=[-2,-1], oshape=mps.shape)

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
max_eigen = None
verbose = False

# Scale b0
b0_scaled = b0 * trj.shape[0] * dt
ts_scaled = (torch.arange(trj.shape[0], device=torch_dev, dtype=torch.float32) + (6.5e-3 / dt)) / trj.shape[0]
phis = torch.cat([phis, b0_scaled[None,]], dim=0)
alphas = torch.cat([alphas, ts_scaled[None, :, None]], dim=0)

# GROG data
hop = high_order_phase(phis, alphas, use_KB=True)
gparams = gridding_params(num_inputs=1,
                          kern_width=2,
                          interp_readout=True,
                          grid=False)
tparams = training_params(epochs=15*2,
                          l2_reg=0.0,
                          loss=lambda x, y : (x - y).abs().square().mean(),
                        #   show_loss=True,
                          float_precision='medium')
grd_data = spatio_temporal_implicit_grogify(ksp, trj, img_cal, hop,
                                            # spatial_funcs=b,
                                            # temporal_funcs=h,
                                            grid_params=gparams,
                                            train_params=tparams,)
ksp = grd_data['ksp_grd']
kerns = grd_data['model'].forward_kernel(grd_data['feats'])
kerns = rearrange(kerns, '... Co Ci -> Co Ci ...')
kerns = kerns.reshape((C, C, *dcf.shape))

# b = einsum(b, bcc, 'L1 ..., L2 ... -> L1 L2 ...').reshape((-1, *im_size))
# h = einsum(h, hcc, 'L1 ..., L2 ... -> L1 L2 ...').reshape((-1, *dcf.shape))

# Spatio-temporal decomposition
# kerns = torch.zeros((C, C, *dcf.shape), dtype=torch.complex64, device=torch_dev)
# tup = (slice(None),)*2 + (None,) * dcf.ndim
# kerns[...] = torch.eye(C, dtype=torch.complex64, device=torch_dev)[tup]
b, h = decomp_with_grappa(hop, kerns, mps, L=50, niter=50)
# b, h = svd_decomp_operator(hop, L=8)

# Apply bases to A and reconstruct
bparams = batching_params(field_batch_size=1,)
A = experimental_sense_coil(trj, b, h, dcf, bparams=bparams)
# A = experimental_sense(trj, mps, dcf, spatial_funcs=b, temporal_funcs=h, bparams=bparams)
img_recon = CG_SENSE_recon(A, ksp, max_iter, lamda_l2, max_eigen, verbose).cpu().T

plt.figure(figsize=(7,7))
plt.imshow(img_recon.abs(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
