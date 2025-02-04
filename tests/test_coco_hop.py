import time
import torch
import sigpy as sp
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase
from mr_recon.imperfections.imperf_decomp import svd_decomp_operator, temporal_segmentation
from mr_recon.algs import density_compensation
from mr_recon.linops import batching_params, sense_linop, experimental_sense
from mr_recon.recons import CG_SENSE_recon, coil_combine, FISTA_recon
from mr_recon.utils import gen_grd, normalize, np_to_torch, torch_to_np, quantize_data
from mr_recon.fourier import fft, ifft, sigpy_nufft, _torch_apodize
from mr_recon.imperfections.main_field import main_field_imperfection
from mr_recon.imperfections.coco import coco_imperfection
from mr_recon.imperfections.combine import combined_imperfections
from mr_recon.imperfections.exponential import exponential_imperfection
from igrog.fixed_kernel_models import learnable_kernels, learnable_kernels_cont, kaiser_bessel_model
from igrog.datasets import general_fixed_dataset
from igrog.training import training_params, stochastic_train_fixed
from igrog.kernel_linop import fixed_kern_linop, fixed_kern_coil_linop
from einops import rearrange, einsum
from tqdm import tqdm

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
                                    L=20,
                                    method='ts',
                                    interp_type='lstsq',
                                    verbose=True)
phis = coco_imperf.phis[:-1] # ignore last term
alphas = coco_imperf.alphas[:-1] # ignore last term

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
# phis = torch.cat([phis, b0_scaled[None,]], dim=0)
# alphas = torch.cat([alphas, ts_scaled[None, :, None]], dim=0)

# HOP
from mr_recon.spatial import spatial_resize
phis = spatial_resize(phis, (100,100), method='bicubic')
hop = high_order_phase(phis, alphas)#, spatial_batch_size=im_size[0] * 20)
# b, h = svd_decomp_operator(hop, L=5, fast_axis='spatial')
b, h = temporal_segmentation(hop, L=20, interp_type='lstsq')
b = spatial_resize(b, im_size, method='bicubic')


bparams = batching_params(C * 0 + 1)
# A = sense_linop(im_size, trj, mps, dcf, 
#                 imperf_model=coco_imperf,
#                 bparams=bparams)
A = experimental_sense(trj, mps, dcf, bparams=bparams, spatial_funcs=b, temporal_funcs=h)
img_recon = CG_SENSE_recon(A, ksp, max_iter, lamda_l2, max_eigen, verbose).cpu()

plt.figure()
plt.imshow(img_recon.abs(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
quit()


# Sweep over L and W
# Ls = torch.arange(1, 15, 2)
# Ws = torch.arange(5, 12, 2)
Ls = torch.tensor([1, 3, 5, 7, 9, 13, 27, 64, 125])
Ws = torch.arange(5, 6, 1)
l, w = torch.meshgrid([Ls, Ws], indexing='ij')
l = l.flatten()
w = w.flatten()
times = torch.zeros(len(w), dtype=torch.float32)
times_ts = torch.zeros(len(w), dtype=torch.float32)
errors = torch.zeros(len(w), dtype=torch.float32)
errors_ts = torch.zeros(len(w), dtype=torch.float32)
for i in tqdm(range(len(w))):
    L = l[i].item()
    W = w[i].item()
    
    # Build source points
    kern_size = (W,W)
    src_vecs = gen_grd(kern_size, kern_size) / os_grd
    src_vecs = src_vecs.reshape(-1, 2).to(torch_dev)
    src_vecs -= src_vecs.mean(dim=0)
    
    # Model features
    kdev = trj - (trj * os_grd).round() / os_grd
    features = kdev.clone()

    # Get coco basis functions
    coco_imperf = coco_imperfection(trj, im_size, fov=(0.22,)*2, dt=dt, B0=3.0, z_ofs=0.0, rotations=(0,)*2,
                                    L=1,
                                    method='ts',
                                    interp_type='zero',
                                    verbose=verbose)
    phis = coco_imperf.phis.moveaxis(0, -1).clone()[..., :-1] # ignore last term
    alphas = coco_imperf.alphas.moveaxis(0, -1).clone()[..., :-1] # ignore last term
    phis = torch.cat([phis, b0_scaled[..., None]], dim=-1)
    alphas = torch.cat([alphas, ts_scaled[:, None, None]], dim=-1)
    inds_seg = quantize_data(torch.arange(trj.shape[0], device=torch_dev)[:, None], L)[0] # L 1
    alpha_clusters = alphas[inds_seg[:, 0], 0, :] 
    
    # Prep source maps
    source_maps = torch.exp(-2j * torch.pi * einsum(phis, alpha_clusters, '... B, L B -> L ... '))
    features = torch.cat([features, alphas], dim=-1)
    basis_funcs = torch.cat([gen_grd(im_size).to(torch_dev),
                            phis], dim=-1)

    # Make dataset
    target_maps = torch.ones((1, *im_size), device=torch_dev, dtype=torch.complex64)
    basis_coeffs = rearrange(features, '... B -> B (...)')
    basis_funcs = rearrange(basis_funcs, '... B -> (...) B')
    kern_model = learnable_kernels(basis_coeffs.shape[0], kern_size, im_size, source_maps, target_maps).to(torch_dev)
    dataset = general_fixed_dataset(src_vecs, kern_model.source_maps, kern_model.target_maps, basis_coeffs, basis_funcs, msk)

    # Train
    tparams.epochs = 1
    save_path = f'/local_mount/space/mayday/data/users/abrahamd/hofft/save_dir/models/features_coco_{L}_{W}.pt'
    # try:
    #     feature_layers = torch.load(save_path)
    #     kern_model.feature_layers = feature_layers
    # except:
    #     pass
    kern_model = stochastic_train_fixed(kern_model, dataset, tparams, verbose=verbose)
    # torch.save(kern_model.feature_layers, save_path)
#     continue
# quit()
    
    if W % 2 == 0:
        k = torch.ones(2, device=torch_dev, dtype=torch.float32) * 0.5 / os_grd
        rs = gen_grd(im_size).to(torch_dev)
        phz = torch.exp(-2j * torch.pi * einsum(rs, k, '... d, d -> ...'))
        kern_model.source_maps.data *= phz

    # Recon
    bparams = batching_params(field_batch_size=10)
    A = fixed_kern_linop(kern_model, mps, trj, features, dcf, os_grd, False, bparams)
    if i == 0:
        img_recon = CG_SENSE_recon(A, ksp, max_iter, lamda_l2, max_eigen, verbose)
    start = time.perf_counter()
    img_recon = CG_SENSE_recon(A, ksp, max_iter, lamda_l2, max_eigen, verbose)
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_kern = end - start
    
    # Compute TS recon
    imperf_model = exponential_imperfection(phis.moveaxis(-1, 0), alphas.moveaxis(-1, 0),
                                            custom_clusters=alpha_clusters,
                                            L=L,
                                            method='ts',
                                            interp_type='zero',
                                            verbose=verbose)
    nufft = sigpy_nufft(im_size, os_grd, W)
    A_reg = sense_linop(im_size, trj, mps, dcf, bparams=bparams, imperf_model=imperf_model, nufft=nufft)
    start = time.perf_counter()
    img_reg = CG_SENSE_recon(A_reg, ksp, max_iter, lamda_l2, max_eigen, verbose)
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_ts = end - start
    
    # Track time and error
    img_reg = normalize(img_reg.cpu(), img_gt)
    img_recon = normalize(img_recon.cpu(), img_gt)
    errors[i] = (img_recon - img_gt).norm() / img_gt.norm()
    errors_ts[i] = (img_reg.abs() - img_gt.abs()).norm() / img_gt.norm()
    times[i] = t_kern
    times_ts[i] = t_ts

# Save stuff
grd_size = (len(Ls), len(Ws))
save_data = {
    'ls': l.reshape(grd_size).numpy(),
    'ws': w.reshape(grd_size).numpy(),
    'times': times.reshape(grd_size).numpy(),
    'times_ts': times_ts.reshape(grd_size).numpy(),
    'errors': errors.reshape(grd_size).numpy(),
    'errors_ts': errors_ts.reshape(grd_size).numpy(),
}
# np.savez('save_dir/err_comp_coco.npz', **save_data)
np.savez('/local_mount/space/mayday/data/users/abrahamd/hofft/save_dir/err_comp_zero_coco.npz', **save_data)
