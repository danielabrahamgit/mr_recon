import torch
import sigpy as sp
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from einops import einsum, rearrange

from mr_recon.linops import experimental_sense, experimental_sense_coil, batching_params
from mr_recon.recons import CG_SENSE_recon, coil_combine
from mr_recon.imperfections.field import (
    b0_to_phis_alphas, 
    coco_to_phis_alphas, 
    alpha_phi_svd, 
    alpha_segementation, 
    alpha_phi_svd_with_grappa
)

from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.fourier import ifft, sigpy_nufft
from mr_recon.utils import gen_grd  
from mr_recon.spatial import spatial_resize

from igrog.kernel_linop import fixed_kern_naive_linop
from igrog.training import training_params
from igrog.grogify import spatio_temporal_implicit_grogify
from igrog.gridding import gridding_params

# Set seeds
torch.manual_seed(0)
np.random.seed(0)

# load data
torch_dev = torch.device(4)
# torch_dev = torch.device('cpu')
R = 3
dt = 2e-6
FOV = 0.22
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
# phi_size = (64,)*2
phi_size = im_size
ksp_cal = synth_cal(ksp_fs, (32, 32), trj_fs, dcf_fs, num_iter=1)
img_cal = ifft(ksp_cal, dim=[-2,-1], oshape=(C, *phi_size))


# Get phis alphas from field imperfections
xyz = gen_grd(im_size).to(torch_dev)
xyz = torch.stack([xyz[..., 0], xyz[..., 0]*0, xyz[..., 1]], dim=-1)
trj_physical = torch.stack([trj[..., 0], trj[..., 0]*0, trj[..., 1]], dim=-1) * FOV
phis_coco, alphas_coco = coco_to_phis_alphas(trj_physical, xyz, 3, 0, dt)
phis_b0, alphas_b0 = b0_to_phis_alphas(b0, dcf.shape, 0, dt)
phis = torch.cat([phis_coco[:-2], phis_b0], dim=0)
alphas = torch.cat([alphas_coco[:-2], alphas_b0], dim=0)
phis = spatial_resize(phis, phi_size, method='bicubic')

# # Add on non-cartesian term
# os = 1.5
# rs = gen_grd(phi_size).to(torch_dev).moveaxis(-1, 0)
# kdevs = (trj - (os * trj).round()/os).moveaxis(-1, 0)
# phis = torch.cat([phis, rs], dim=0)
# alphas = torch.cat([alphas, kdevs], dim=0)

# GROG data
# hop = high_order_phase(phis, alphas, use_KB=False, temporal_batch_size=2**10)
# gparams = gridding_params(num_inputs=1,
#                           kern_width=2,
#                           interp_readout=False,
#                           grid=False)
# tparams = training_params(epochs=15*0+1,
#                           l2_reg=0.0,
#                           loss=lambda x, y : (x - y).abs().square().mean(),
#                         #   show_loss=True,
#                           float_precision='medium')
# grd_data = spatio_temporal_implicit_grogify(ksp, trj, img_cal, hop,
#                                             # spatial_funcs=b,
#                                             # temporal_funcs=h,
#                                             grid_params=gparams,
#                                             train_params=tparams,)
# ksp = grd_data['ksp_grd']
# kerns = grd_data['model'].forward_kernel(grd_data['feats'])
# kerns = rearrange(kerns, '... Co Ci -> Co Ci ...')
# kerns = kerns.reshape((C, C, *dcf.shape))

# b = einsum(b, bcc, 'L1 ..., L2 ... -> L1 L2 ...').reshape((-1, *im_size))
# h = einsum(h, hcc, 'L1 ..., L2 ... -> L1 L2 ...').reshape((-1, *dcf.shape))

# Spatio-temporal decomposition
# TODO Debug 
L = 4
LC = C * L
kerns = torch.zeros((C, C, *dcf.shape), dtype=torch.complex64, device=torch_dev)
tup = (slice(None),)*2 + (None,) * dcf.ndim
kerns[...] = torch.eye(C, dtype=torch.complex64, device=torch_dev)[tup]
b, h = alpha_phi_svd_with_grappa(phis, alphas, kerns, mps, 
                                 L=LC, L_batch_size=LC, use_type3=True)
b /= coil_combine(img_cal)
# b, h = alpha_phi_svd(phis, alphas, L=L, L_batch_size=L)
b = spatial_resize(b, im_size, method='bicubic')

# Apply bases to A and reconstruct
bparams = batching_params(field_batch_size=1,)
A = experimental_sense_coil(trj, b, h, dcf, bparams=bparams)
# A = experimental_sense(trj, mps, dcf, spatial_funcs=b, temporal_funcs=h, bparams=bparams)
img_recon = CG_SENSE_recon(A, ksp, 10, 0, 1.0).cpu().rot90()

plt.figure(figsize=(7,7))
plt.imshow(img_recon.abs(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
