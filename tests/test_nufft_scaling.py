import torch
import numpy as np

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import np_to_torch, torch_to_np, gen_grd
from einops import rearrange, einsum
from mr_recon.fourier import (
    chebyshev_nufft, 
    cufi_nufft, 
    matrix_nufft, 
    sigpy_nufft, 
    svd_nufft, 
    torchkb_nufft, 
    gridded_nufft, 
    triton_nufft
)

# # Random seeds
# np.random.seed(0)
# torch.manual_seed(0)

# Params
device_idx = 0
grd_os = round(1.333 * 294) / 294
try:
    torch_dev = torch.device(device_idx)
except:
    torch_dev = torch.device('cpu')
d = 3
im_size = (12, 294, 150)
rtol = 5e-2 
eps = 1e-6

# Make grid and k-space coordinates
grd = gen_grd(im_size).to(torch_dev)
crds = (torch.rand((1000, d), dtype=torch.float32, device=torch_dev) - 0.5)
for i in range(d):
    crds[..., i] *= im_size[i]
crds = torch.round(crds * grd_os) / grd_os
# img = np_to_torch(sp.shepp_logan(im_size)).to(torch_dev).type(torch.complex64)
img = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)

# Create nuffts
nuffts = [
    gridded_nufft(im_size, grd_os), 
    sigpy_nufft(im_size, oversamp=2.0),
]
names = [str(nufft) for nufft in nuffts]
def plot_err_ksp(err):
    plt.plot(err.cpu())
    plt.ylim(-.2, 1.2)
    plt.show()
def plot_err_img(err):
    plt.imshow(err.cpu())
    plt.show()

# ------------- Test Forward ---------------
kern = torch.exp(-2j * torch.pi * einsum(crds, grd, 'n two, ... two -> n ...'))
ksp_dft = einsum(kern, img, 'n ..., ... -> n') / np.sqrt(np.prod(im_size))
for nufft, name in zip(nuffts, names):
    crds_rs = nufft.rescale_trajectory(crds)
    ksp = nufft(img[None,], crds_rs[None,])[0]
    err = (ksp - ksp_dft).abs() / (ksp_dft.abs() + eps)
    assert torch.all(err <= rtol), f'{name} forward failed {plot_err_ksp(err)}'

# ------------- Test Adjoint ---------------
ksp = ksp_dft.clone()
img_dft = einsum(kern.conj(), ksp, 'n ... , n -> ...') / np.sqrt(np.prod(im_size))
for nufft, name in zip(nuffts, names):
    crds_rs = nufft.rescale_trajectory(crds)
    img_adj = nufft.adjoint(ksp[None,], crds_rs[None,])[0]
    err = (img_adj - img_dft).abs() / (img_dft.abs() + eps)
    assert torch.all(err <= rtol), f'{name} adjoint failed {plot_err_img(err)}'

# ------------- Test Toeplitz ---------------
aha_dft = einsum(kern.conj(), ksp_dft, 'n ... , n -> ...') / np.sqrt(np.prod(im_size))
for nufft, name in zip(nuffts, names):
    crds_rs = nufft.rescale_trajectory(crds)
    kern = nufft.calc_teoplitz_kernels(crds_rs[None,])
    aha = nufft.normal_toeplitz(img[None, None], kern)[0,0]
    err = (aha - aha_dft).abs() / (aha_dft.abs() + eps)
    assert torch.all(err <= rtol), f'{name} toeplitz failed {plot_err_img(err)}'

# ------------- Test Toeplitz Compared to no Toeplitz ---------------
for nufft, name in zip(nuffts, names):
    crds_rs = nufft.rescale_trajectory(crds)

    # Forward backward approach
    frwrd = nufft(img[None,], crds_rs[None,])[0]
    aha_no = nufft.adjoint(frwrd[None,], crds_rs[None,])[0]

    # Toeplitz approach
    kern = nufft.calc_teoplitz_kernels(crds_rs[None,])
    aha = nufft.normal_toeplitz(img[None, None], kern)[0,0]

    err = (aha - aha_no).abs() / (aha_no.abs() + eps)
    assert torch.all(err <= rtol), f'{name} toeplitz - normal failed {plot_err_img(err)}'
    
print('All tests passed')