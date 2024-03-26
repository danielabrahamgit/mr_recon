import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import sigpy as sp
from mr_recon.fourier import sigpy_nufft, torchkb_nufft, gridded_nufft
from mr_recon.utils import np_to_torch, torch_to_np, gen_grd
from einops import rearrange, einsum

# Random seeds
np.random.seed(0)
torch.manual_seed(0)

# Params
device_idx = 6 * 0 - 1
grd_os = 2.0
try:
    torch_dev = torch.device(device_idx)
except:
    torch_dev = torch.device('cpu')
im_size = (220, 220)

# Make grid and k-space coordinates
grd = gen_grd(im_size).to(torch_dev)
crds = (torch.rand((4, 2), dtype=torch.float32, device=torch_dev) - 0.5)
crds[..., 0] *= im_size[0] * 0.99
crds[..., 1] *= im_size[1] * 0.99
crds = torch.round(crds * grd_os) / grd_os
img = np_to_torch(sp.shepp_logan(im_size)).to(torch_dev).type(torch.complex64)

# ------------- Test Forward ---------------
# DFT 
kern = torch.exp(-2j * torch.pi * einsum(crds, grd, 'n two, ... two -> n ...'))
ksp_dft = einsum(kern, img, 'n ..., ... -> n') / np.sqrt(np.prod(im_size))

# kb NUFFT
kb_nufft = torchkb_nufft(im_size, device_idx=device_idx)
crds_kb = kb_nufft.rescale_trajectory(crds)
ksp_kb = kb_nufft(img[None,], crds_kb[None,])[0]

# sigpy NUFFt
sp_nufft = sigpy_nufft(im_size, device_idx=device_idx, oversamp=1.5, width=6)
crds_sp = sp_nufft.rescale_trajectory(crds)
ksp_sp = sp_nufft(img[None,], crds_sp[None,])[0]

# gridded NUFFt
grd_nufft = gridded_nufft(im_size, device_idx=device_idx, grid_oversamp=grd_os)
crds_grd = grd_nufft.rescale_trajectory(crds)
ksp_grd = grd_nufft(img[None,], crds_grd[None,])[0]

# Compare
assert torch.allclose(ksp_dft, ksp_kb, atol=0.0, rtol=1e-2)
assert torch.allclose(ksp_dft, ksp_sp, atol=0.0, rtol=1e-2)
assert torch.allclose(ksp_dft, ksp_grd, atol=0.0, rtol=1e-2)

# ------------- Test Adjoint ---------------
# iDFT
ksp = torch.zeros_like(ksp_dft)
ksp[2] = 1
img_dft = einsum(kern.conj(), ksp, 'n ... , n -> ...') / np.sqrt(np.prod(im_size))

# kb NUFFT
img_kb = kb_nufft.adjoint(ksp[None,], crds_kb[None,])[0]

# sigpy NUFFT
img_sp = sp_nufft.adjoint(ksp[None,], crds_sp[None,])[0]

# gridded NUFFT
img_grd = grd_nufft.adjoint(ksp[None,], crds_grd[None,])[0]

# Compare
assert torch.allclose(img_dft, img_kb, atol=0.0, rtol=1e-2)
assert torch.allclose(img_dft, img_sp, atol=0.0, rtol=1e-2)
assert torch.allclose(img_dft, img_grd, atol=0.0, rtol=1e-2)

# ------------- Test Toeplitz ---------------
# DFT
aha_dft = einsum(kern.conj(), ksp_dft, 'n ... , n -> ...') / np.sqrt(np.prod(im_size))

# kb NUFFT
kern_kb = kb_nufft.calc_teoplitz_kernels(crds_kb[None,])
aha_kb = kb_nufft.normal_toeplitz(img[None, None], kern_kb)[0,0]

# sigpy NUFFT
kern_sp = sp_nufft.calc_teoplitz_kernels(crds_sp[None,])
aha_sp = sp_nufft.normal_toeplitz(img[None, None], kern_sp)[0,0]

# gridded NUFFT
kern_grd = grd_nufft.calc_teoplitz_kernels(crds_grd[None,], os_factor=grd_os)
aha_grd = grd_nufft.normal_toeplitz(img[None, None], kern_grd)[0,0]

# Compare
assert torch.allclose(aha_dft, aha_kb, atol=0.0, rtol=1e-2)
assert torch.allclose(aha_dft, aha_sp, atol=0.0, rtol=1e-2)
assert torch.allclose(aha_dft, aha_grd, atol=0.0, rtol=1e-2), torch.norm(aha_dft) / torch.norm(aha_grd)

# ------------- Test Toeplitz Compared to no Toeplitz ---------------
nuffts = [kb_nufft, sp_nufft, grd_nufft]
for nufft in nuffts:

    # rescale
    crds_rs = nufft.rescale_trajectory(crds)

    # Forward backward approach
    frwrd = nufft(img[None,], crds_rs[None,])[0]
    aha_no = nufft.adjoint(frwrd[None,], crds_rs[None,])[0]

    # Toeplitz approach
    kern = nufft.calc_teoplitz_kernels(crds_rs[None,])
    aha = nufft.normal_toeplitz(img[None, None], kern)[0,0]

    assert torch.allclose(aha, aha_no, atol=0.0, rtol=1e-2), f'{nufft} failed'
plt.show()