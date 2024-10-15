import torch
import sigpy as sp
import torch.nn.functional as F

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from mr_recon.fourier import fft, ifft
from mr_recon.utils import np_to_torch
from sigpy.mri import birdcage_maps
from einops import rearrange

# ------------------ Parmeters ------------------
C = 12 # number of channels
R = 2 # acceleration factor
cal_size = (32, 32) # calibration region size
im_size = (256, 256) # image dimensions as (Nx, Ny)
kern_size = (5, 2) # GRAPPA kernel size
lamda_l2 = 1e-2*0 # GRAPPA L2 regularization parameter
torch_dev = torch.device(5) # GPU device

# ------------------ Simulate ------------------
# Make fake image and sensitivity maps to use 
img = np_to_torch(sp.shepp_logan(im_size)).type(torch.complex64).to(torch_dev)
mps = np_to_torch(birdcage_maps((C, *im_size), r=1)).type(torch.complex64).to(torch_dev)

# Simulate calibration data
ksp_cal = fft(img * mps, dim=[-2,-1])[:, im_size[0]//2 - cal_size[0]//2:im_size[0]//2 + cal_size[0]//2, 
                                         im_size[1]//2 - cal_size[1]//2:im_size[1]//2 + cal_size[1]//2]

# Simulate k-space data, undersample the Y dimension
ksp = fft(img * mps, dim=[-2, -1])[:, :, ::R]

# ------------------ Train GRAPPA kernel ------------------
# Extract fully sampled (fs) patches
patch_size_fs = (kern_size[0], kern_size[1] * R)
patch_fs = F.unfold(ksp_cal[None], patch_size_fs)[0]
patch_fs = rearrange(patch_fs, '(C Kx Ky) N -> N C Kx Ky',
                     C=C, Kx=patch_size_fs[0], Ky=patch_size_fs[1])

# Undersample kernel
source = rearrange(patch_fs[..., 1::R], 'N C Kx Ky_us -> N (C Kx Ky_us)') # N (C * Ksize)
target = patch_fs[..., patch_size_fs[0]//2, patch_size_fs[1]//2] # N C

# Solve for GRAPPA kernel
SHS = source.H @ source
SHT = source.H @ target
I = torch.eye(SHS.shape[-1], device=torch_dev, dtype=torch.complex64)
grappa_kernel = torch.linalg.solve(SHS + lamda_l2 * I, SHT)

# ------------------ Apply GRAPPA kernel ------------------
# Extract undersampled points
source = F.unfold(ksp[None], kern_size)[0]
source = rearrange(source, '(C Kx Ky) N -> N (C Kx Ky)',
                   C=C, Kx=kern_size[0], Ky=kern_size[1])

# Apply GRAPPA Kernel
target = (source @ grappa_kernel).T
target = target.reshape((C, im_size[0] - 2*(kern_size[0]//2), im_size[1]//R - kern_size[1]//2))

# Insert into kspace matrix
ksp_grappa = torch.zeros((C, *im_size), device=torch_dev, dtype=torch.complex64)
ksp_zero_filled = ksp_grappa.clone()
ksp_zero_filled[..., ::R] = ksp
ksp_grappa[..., ::R] = ksp
ksp_grappa[:, (kern_size[0]//2):-(kern_size[0]//2), 1:im_size[1]-1:R] = target

# iFFT Recon to image and square root sum squares over coil dimension
img_grappa = ifft(ksp_grappa, dim=[-2, -1]).abs().square().sum(dim=0).sqrt()
img_zero_filled = ifft(ksp_zero_filled, dim=[-2, -1]).abs().square().sum(dim=0).sqrt()

# ------------------ Plot ------------------
vmin = 0
vmax = img.abs().max().item()
plt.figure(figsize=(14,7))
plt.subplot(131)
plt.title(f'Ground Truth')
plt.imshow(img.abs().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(132)
plt.title(f'Naive Recon (zero filled)')
plt.imshow(img_zero_filled.cpu(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(133)
plt.title(f'GRAPPA Recon')
plt.imshow(img_grappa.cpu(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.tight_layout()
plt.show()
