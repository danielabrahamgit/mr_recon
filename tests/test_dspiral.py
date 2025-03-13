import time
import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

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
db0 = (1_000 - (hop_eddy.phis[3] + hop_eddy.phis[4] - hop_eddy.phis[8])) * (mps[0].abs() > 0)
db0 = 25 * db0 / db0.abs().max()
a_b0, p_b0 = alphas_phis_from_B0(b0 + db0, trj.shape[:-1], dt)
hop_b0 = high_order_phase(p_b0, a_b0, 
                          use_KB=False, temporal_batch_size=2**4)
b0_segs = temporal_segmentation(hop_b0, L=20, interp_type='lstsq')
A_b0 = experimental_sense(trj, mps, dcf, 
                          spatial_funcs=b0_segs[0],
                          temporal_funcs=b0_segs[1],
                          bparams=bparams)

# Both
hop = high_order_phase(torch.cat([p_b0, p_eddy], dim=0), 
                       torch.cat([a_b0, a_eddy], dim=0), 
                       use_KB=False, temporal_batch_size=2**4)
segs = temporal_segmentation(hop, L=30, interp_type='lstsq')
A_both = experimental_sense(trj, mps, dcf, 
                            spatial_funcs=segs[0],
                            temporal_funcs=segs[1],
                            bparams=bparams)
# ----------------------------------------------------

# Auto-focus center of k-space 
recon = lambda k, A : CG_SENSE_recon(A, k, max_iter=7, lamda_l2=0.0, max_eigen=1.0, verbose=False)
freqs = torch.linspace(-db0.abs().max(), db0.abs().max(), 11, device=torch_dev)
imgs = []
imgs_lowres = []
t = torch.arange(0, trj.shape[0], device=torch_dev) * dt
for freq in tqdm(freqs, 'Multi-Freq Recons'):
    phz = torch.exp(-2j * torch.pi * freq * t)
    imgs.append(recon(ksp * phz, A_both))
    ro_inds = slice(0, 2_000)
    A_lowres = experimental_sense(trj[ro_inds], mps, dcf[ro_inds], 
                            spatial_funcs=segs[0],
                            temporal_funcs=segs[1][:, ro_inds],
                            bparams=bparams)
    imgs_lowres.append(recon((ksp * phz)[:, ro_inds], A_lowres))
imgs = torch.stack(imgs, dim=0)
imgs_lowres = torch.stack(imgs_lowres, dim=0)

# Estimate field map
imgs_pc = imgs * torch.exp(-1j * imgs_lowres.angle())
metric = imgs_pc.imag.abs()
W = 15
d = 2
metric = torch.nn.functional.conv2d(
        metric[:, None],
        torch.ones(1, 1, *([W] * d), device=torch_dev) / W**d, 
        padding=W // 2
    ).view(imgs.shape[0], *im_size).abs()
idxs = torch.argmin(metric, dim=0)
field_map = freqs[idxs]

# from pyeyes import ComparativeViewer
# cv = ComparativeViewer({'field_map': field_map.cpu(), 'db0': db0.cpu()},
#                        ['X', 'Y'], ['X', 'Y'])
# cv.launch()
# quit()

# Show
plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.imshow(field_map.cpu(), cmap='jet', vmin=freqs.min(), vmax=freqs.max())
plt.axis('off')
plt.subplot(122)
plt.imshow(db0.cpu(), cmap='jet', vmin=freqs.min(), vmax=freqs.max())
plt.axis('off')
plt.tight_layout()

for i in range(len(freqs)):
    plt.figure(figsize=(14, 7))
    plt.suptitle(f'Freq = {freqs[i].item():.2f}Hz')
    plt.subplot(131)
    plt.imshow(imgs[i].abs().cpu(), cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(imgs_pc[i].angle().cpu(), cmap='jet', vmin=-np.pi, vmax=np.pi)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(metric[i].abs().cpu(), vmin=0, vmax=metric.max())
    plt.axis('off')
    plt.tight_layout()
plt.show()
