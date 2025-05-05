import torch
import sigpy as sp
import torch.nn.functional as F

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from mr_recon.spatial import spatial_resize
from mr_recon.fourier import fft, ifft, gridded_nufft, sigpy_nufft
from mr_recon.utils import np_to_torch, gen_grd, quantize_data, resize, normalize
from mr_recon.recons import coil_combine
from sigpy.mri import birdcage_maps
from einops import rearrange
from scipy.ndimage import gaussian_filter

from igrog.gridding import choose_source_target_greedy, reorder_source, gridding_params
from igrog.kernel_models import brute_force_grappa, implicit_grog
from igrog.training import stochastic_train, training_params
from igrog.datasets import gridding_dataset

# ------------------ Parmeters ------------------
C = 12 # number of channels
R = 2 # acceleration factor
cal_size = (40, 40) # calibration region size
im_size = (256, 256) # image dimensions as (Nx, Ny)
kern_size = (5, 2) # GRAPPA kernel size
lamda_l2 = 1e-2*0 # GRAPPA L2 regularization parameter
# torch_dev = torch.device(4) # GPU device
torch_dev = torch.device('cpu') # CPU device

# ------------------ Simulate ------------------
# Make fake image and sensitivity maps to use 
img = np_to_torch(sp.shepp_logan(im_size)).type(torch.complex64).to(torch_dev)
mps = np_to_torch(birdcage_maps((C, *im_size), r=1)).type(torch.complex64).to(torch_dev)

# Simulate calibration data
ksp_cal = fft(img * mps, dim=[-2,-1])[:, im_size[0]//2 - cal_size[0]//2:im_size[0]//2 + cal_size[0]//2, 
                                         im_size[1]//2 - cal_size[1]//2:im_size[1]//2 + cal_size[1]//2]
ksp_cal /= ksp_cal.abs().max()
img_cal = ifft(ksp_cal, dim=[-2, -1])

# Simulate k-space data
ksp_fs = fft(img * mps, dim=[-2, -1])

# Undersample
W = 10 * 2
im_size_W = (im_size[0] + W, im_size[1] + W)
mask = torch.ones((im_size[0], im_size[1]), device=torch_dev, dtype=torch.complex64)
mask[:, ::R] = 0
mask = 1 - resize(mask, im_size_W)
ksp_fs = resize(ksp_fs, (C, *im_size_W))
im_size_tensor_W = torch.tensor(im_size_W, device=torch_dev)
im_size_tensor = torch.tensor(im_size, device=torch_dev)
inds_acq = torch.argwhere(mask == 1)
inds_trg = torch.argwhere(mask == 0)
trj_acq = (inds_acq - im_size_tensor_W // 2).float()
trj_trg = (inds_trg - im_size_tensor_W // 2).float()
ksp_acq = ksp_fs[:, inds_acq[:, 0], inds_acq[:, 1]]

# ------------------ GRAPPA ------------------
# Source points and vectors
S = 10
src_inds, src_vecs = choose_source_target_greedy(trj_acq[:, None], trj_trg[:, None], radius=3, 
                                                 num_neighbors=S, num_src=S, random=False)
src_vecs = src_vecs[:, 0]
src_inds = src_inds[:, 0]
src_inds, src_vecs = reorder_source(src_inds, src_vecs, method='grid')

# clust, idx = quantize_data(src_vecs.reshape((-1, S * 2)), 1, 'unique')
# clust = clust.reshape((clust.shape[0], S, 2))
# clust_mode = clust[idx.mode().values]

# # Train
# from mr_recon.multi_coil.grappa_est import grappa_AHA_AHb, grappa_AHA_AHb_fast, grappa_AHA_AHb_img
# AHA, AHb = grappa_AHA_AHb(img_cal, clust_mode[None,])
# AHA = AHA.cpu()
# AHb = AHb.cpu()
# from mr_recon.spatial import spatial_resize
# AHA_img, AHb_img = grappa_AHA_AHb_img(spatial_resize(img_cal, (24, 24), method='fourier'), clust_mode[None,])
# AHA_img = normalize(AHA_img.cpu(), AHA, mag=True, ofs=False)
# AHb_img = normalize(AHb_img.cpu(), AHb, mag=True, ofs=False)

# plt.figure(figsize=(14, 7))
# plt.subplot(131)
# plt.title('AHA')
# plt.imshow(AHA[0].abs())
# plt.colorbar()
# plt.axis('off')
# plt.subplot(132)
# plt.title('AHA_img')
# plt.imshow(AHA_img[0].abs())
# plt.colorbar()
# plt.axis('off')
# plt.subplot(133)
# plt.title('Diff')
# plt.imshow((AHA_img[0] - AHA[0]).abs())
# plt.colorbar()
# plt.axis('off')
# plt.tight_layout()
# plt.show()
# quit()
# breakpoint()

# Train Kernels
img_cal /= img_cal.abs().max()
kern_model = brute_force_grappa(img_cal, S, lamda_tikonov=1e-5, solver='solve')
# tparams = training_params(30, loss=lambda x, y: (x - y).abs().square().sum())
# ds = gridding_dataset(img_cal, src_vecs)
# kern_model = implicit_grog(S*2, S, C, C).to(torch_dev)
# kern_model = stochastic_train(kern_model, ds, tparams, verbose=True)

# Apply Kernels
ksp_src = rearrange(ksp_acq[:, src_inds], 'C T S -> T S C')
ksp_trg = kern_model(src_vecs.reshape((-1, S*2)), ksp_src)[0].T

# Recon
ksp_cat = torch.cat([ksp_acq, ksp_trg], dim=1)
trj_cat = torch.cat([trj_acq, trj_trg], dim=0)
nufft = gridded_nufft(im_size)
nufft = sigpy_nufft(im_size)
# trj_cat = nufft.rescale
img_grappa = nufft.adjoint(ksp_cat[None], trj_cat[None])[0]
img_zero_filled = nufft.adjoint(ksp_acq[None], trj_acq[None])[0]
img_grappa = coil_combine(img_grappa, mps).cpu()
img_zero_filled = coil_combine(img_zero_filled, mps).cpu()
img = img.cpu()
img_grappa = normalize(img_grappa, img)
img_zero_filled = normalize(img_zero_filled, img)

# ------------------ Plot ------------------
vmin = 0
vmax = img.abs().median() + 3 * img.abs().std()
M = 10
plt.figure(figsize=(14,7))
plt.subplot(231)
plt.title(f'Ground Truth')
plt.imshow(img.abs(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(232)
plt.title(f'Naive Recon (zero filled)')
plt.imshow(img_zero_filled.abs(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(233)
plt.title(f'GRAPPA Recon')
plt.imshow(img_grappa.abs(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(234)
plt.imshow((img - img).abs(), cmap='gray', vmin=vmin, vmax=vmax/M)
plt.axis('off')
plt.subplot(235)
plt.imshow((img_zero_filled - img).abs(), cmap='gray', vmin=vmin, vmax=vmax/M)
plt.axis('off')
plt.subplot(236)
plt.imshow((img_grappa - img).abs(), cmap='gray', vmin=vmin, vmax=vmax/M)
plt.axis('off')
plt.tight_layout()
plt.show()
