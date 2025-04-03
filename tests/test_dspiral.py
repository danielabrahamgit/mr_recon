import time
import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.imperfections.estimation.eddy_focus import build_alpha_trajectory, normalize_phis, lowpass_filter_torch
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd, np_to_torch, torch_to_np
from mr_recon.spatial import spatial_resize
from mr_recon.fourier import ifft
from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase, B0, alphas_phis_from_B0, phis_from_spha
from mr_recon.imperfections.imperf_decomp import temporal_segmentation, svd_decomp_matrix, svd_decomp_operator
from mr_recon.linops import batching_params, experimental_sense
from mr_recon.calib import synth_cal

from tqdm import tqdm
from einops import einsum, rearrange
from scipy.ndimage import convolve

# Set seeds
np.random.seed(0)
torch.manual_seed(0)

# GPU stuff
torch_dev = torch.device(4)
# torch_dev = torch.device('cpu')
    
# Recon Stuff
fovs = (0.22,)*2
im_size = (230,)*2
use_toeplitz = False
max_iter = 5
lamda_l2 = 1e-2 
dt = 2e-6
R = 1

# Load data
slc = -5
pth = '/local_mount/space/mayday/data/users/zachs/share/ForDaniel/20241123_Spiral_diffusion_data/'
mps = np_to_torch(np.load(pth + 'mps.npy', mmap_mode='r+')[..., slc]).to(torch_dev).type(torch.complex64)
ksp = np_to_torch(np.load(pth + 'ksp.npy', mmap_mode='r+')[:, :, slc]).to(torch_dev).type(torch.complex64)
alphas = np_to_torch(np.load(pth + 'skopeCoeffs.npy')).to(torch_dev).type(torch.float32)
trj = np_to_torch(np.load(pth + 'trj.npy', mmap_mode='r+')[slc]).to(torch_dev).type(torch.float32)
dcf = np_to_torch(np.load(pth + 'dcf.npy', mmap_mode='r+')[slc]).to(torch_dev).type(torch.float32)
b0 = np_to_torch(np.load(pth + 'b0.npy', mmap_mode='r+')[..., slc]).to(torch_dev).type(torch.float32)
zpos = np.load(pth + 'zpos.npy', mmap_mode='r+')[slc].item()

# Reshape things
dff = 1
ksp = rearrange(ksp, 'C D R one -> C R D one')[..., dff, 0]
alphas = rearrange(alphas, 'D R one B -> B R D one')[..., dff, 0]
trj = trj[:, 0]
dcf = dcf[:, 0]

# Print Shapes
print(f'mps shape = {mps.shape}')
print(f'ksp shape = {ksp.shape}')
print(f'alphas shape = {alphas.shape}')
print(f'trj shape = {trj.shape}')
print(f'dcf shape = {dcf.shape}')
print(f'b0 shape = {b0.shape}')

# -------------- Build segmented models --------------
# Eddy
bparams = batching_params(mps.shape[0])
skp_inds = [0, 1, 2, 4, 6, 8, 9, 11, 13, 15]
xyz = gen_grd(im_size, fovs).to(torch_dev)
xyz = torch.cat([xyz, torch.zeros_like(xyz[..., :1])], dim=-1)
p_eddy = phis_from_spha(xyz, skp_inds)
a_eddy = alphas[skp_inds] / (2 * np.pi)
hop_eddy = high_order_phase(p_eddy, a_eddy, 
                            use_KB=False, temporal_batch_size=2**4)
eddy_segs = temporal_segmentation(hop_eddy, L=20, interp_type='lstsq')
A_eddy = experimental_sense(trj, mps, dcf, 
                            spatial_funcs=eddy_segs[0],
                            temporal_funcs=eddy_segs[1],
                            bparams=bparams)

# B0
a_b0, p_b0 = alphas_phis_from_B0(b0, trj.shape[:-1], dt)
hop_b0 = high_order_phase(p_b0, a_b0, 
                          use_KB=False, temporal_batch_size=2**4)
b0_segs = temporal_segmentation(hop_b0, L=20, interp_type='lstsq')
A_b0 = experimental_sense(trj, mps, dcf, 
                          spatial_funcs=b0_segs[0],
                          temporal_funcs=b0_segs[1],
                          bparams=bparams)
# ----------------------------------------------------


# -------------- Estimate Eddy Currents --------------
# Recon with just b0
img_recon = CG_SENSE_recon(A_b0, ksp, max_iter, lamda_l2, 1.0)

# Est background phase and remove it
img_recon = spatial_resize(img_recon, (32, 32), method='fourier')
img_recon = spatial_resize(img_recon, im_size, method='fourier')
mps = mps * torch.exp(1j * img_recon.angle())
A_b0.mps = mps
img_recon = CG_SENSE_recon(A_b0, ksp, max_iter, lamda_l2, 1.0)

# phis = normalize_phis(p_eddy.clone())
phi_scales = p_eddy.abs().view(p_eddy.shape[0], -1).max(dim=-1).values * 2 # B 
p_eddy /= phi_scales[:, None, None]
a_eddy *= phi_scales[:, None]
# alphas_recon = a_eddy.clone()
alphas_recon = build_alpha_trajectory(img_recon,
                                       p_eddy,
                                       trj,
                                       window_size=500,
                                       num_guesses=50,
                                       noise_scale=.1).T

# Lowpass filter the result to 15e3
alphas_recon = lowpass_filter_torch(alphas_recon, 1e3, fs=1/dt, dim=1, kernel_size=101)

# Show the estimate of alphas
for i in range(a_eddy.shape[0]):
    plt.figure()
    plt.title(f'sph coeff {i}')
    plt.plot(alphas_recon[i].cpu())
    plt.plot(a_eddy[i].cpu())
# ----------------------------------------------------

# -------------- Recon and plot --------------
hop = high_order_phase(torch.cat([p_b0, p_eddy], dim=0), 
                       torch.cat([a_b0, alphas_recon], dim=0), 
                       use_KB=False, temporal_batch_size=2**4)
segs = temporal_segmentation(hop, L=30, interp_type='lstsq')
A_both = experimental_sense(trj, mps, dcf, 
                            spatial_funcs=segs[0],
                            temporal_funcs=segs[1],
                            bparams=bparams)
img_recon = CG_SENSE_recon(A_both, ksp, max_iter, lamda_l2, 1.0).cpu()
img_recon = spatial_resize(img_recon, (500, 500), method='fourier')

plt.figure(figsize=(7, 7))
plt.imshow(img_recon.abs().cpu(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
# ----------------------------------------------------