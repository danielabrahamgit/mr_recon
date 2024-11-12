import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from mr_recon.recons import coil_combine
from mr_recon.multi_coil.coil_est import csm_from_espirit
from mr_recon.fourier import ifft
from mr_recon.utils import np_to_torch
from einops import rearrange

# ------------------ Parmeters ------------------
C = 16 # number of channels
R = 2 # acceleration factor
cal_size = (38, 38) # calibration region size
im_size = (146, 146) # image dimensions as (Nx, Ny)
kern_size = (5, 2) # GRAPPA kernel size
lamda_l2 = 1e-5  # GRAPPA L2 regularization parameter
torch_dev = torch.device(5) # GPU device

# ------------------ Load Data ------------------
# Load kspace data
import numpy as np
ksp_zf = np.load('/local_mount/space/tiger/1/users/abrahamd/kspc_data.npy')
ksp_cal = np.load('/local_mount/space/tiger/1/users/abrahamd/kspc_calib.npy')
ksp_zf = np_to_torch(rearrange(ksp_zf, 'R P C -> C R P')).to(torch_dev)
ksp = ksp_zf[:, :, 35::2]
ksp_cal = np_to_torch(rearrange(ksp_cal, 'R P C -> C R P')).to(torch_dev)
tup = tuple([slice(ksp_cal.shape[i+1]//2 - cal_size[i]//2, ksp_cal.shape[i+1]//2 + cal_size[i]//2) for i in range(2)])
ksp_cal = ksp_cal[(slice(None),) + tup]

# Est maps via espirit on calib
mps = csm_from_espirit(ksp_cal, im_size)[0]

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
target = target.reshape((C, im_size[0] - 2*(kern_size[0]//2), -1))

# Insert into kspace matrix
ksp_grappa = torch.zeros((C, *im_size), device=torch_dev, dtype=torch.complex64)
# ksp_zero_filled = ksp_grappa.clone()
# ksp_zero_filled[..., ::R] = ksp
ksp_zero_filled = ksp_zf.clone()
ksp_grappa = ksp_zf.clone()
ksp_grappa[:, (kern_size[0]//2):-(kern_size[0]//2), 36::2] = target

# iFFT Recon to image and square root sum squares over coil dimension
img_grappa = coil_combine(ifft(ksp_grappa, dim=[-2, -1]), mps).abs()
img_zero_filled = coil_combine(ifft(ksp_zero_filled, dim=[-2, -1]))

# ------------------ Plot ------------------
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.title(f'Naive Recon (zero filled)')
plt.imshow(img_zero_filled.cpu(), cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.title(f'GRAPPA Recon')
plt.imshow(img_grappa.cpu(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
