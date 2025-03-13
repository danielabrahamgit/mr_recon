import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan 
from mr_recon.utils import gen_grd, normalize, np_to_torch
from mr_recon.fourier import sigpy_nufft, fft, ifft
from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase
from mr_recon.imperfections.imperf_decomp import svd_decomp_operator
from einops import einsum


# ------------------- Parameters & Setup -------------------
D = 2 # Dimension of image
B = 2 # Number of basis phases
N = 220 # Number of signal points in each dim
M = 1000 # Total number of transform points
sl = shepp_logan()
# torch_dev = torch.device('cpu')
torch_dev = torch.device(5)

# Define the function and non-linear phases
def f(r):
    # x = r[..., 0]
    # y = r[..., 1]
    # ret = x * 0 + 1
    # ret = torch.cos(2 * torch.pi * 50 * x ** 2)
    ret = sl.img(im_size).to(torch_dev)
    return ret.type(torch.complex64)
def phi(r):
    return torch.stack([r[..., 0] ** 2, r[..., 1] ** 2], dim=-1)
    # return torch.stack([r[..., 0], r[..., 0] ** 2], dim=-1)
    # return r[..., 0:1] ** 2 +  r[..., 1:2] ** 2

# Gen data
im_size = (N,)*D
rs = gen_grd(im_size).to(torch_dev)
fr = f(rs)
phis = phi(rs)
alphas = torch.linspace(-10, 10, M)[:, None].repeat_interleave(B, -1).to(torch_dev)

# ------------------- Naive High Order Transform -------------------
import time
def eval_naive(fr, alphas):
    phase = einsum(alphas, phis, 'M B, ... B -> M ...')
    F_naive = einsum(fr, torch.exp(-2j * torch.pi * phase), '..., M ... -> M')
    return F_naive
def adj_naive(F, alphas):
    phase = einsum(alphas, phis, 'M B, ... B -> M ...')
    fr = einsum(F, torch.exp(2j * torch.pi * phase), 'M, M ... -> ...')
    return fr
start = time.perf_counter()
F_naive = eval_naive(fr, alphas)
# F_naive = adj_naive(F_naive, alphas)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Elapsed time: {end - start}')
F_naive = F_naive.cpu()

# -------------------- HOP Library Transform -------------------
phi_hop = phis.moveaxis(-1, 0)
alphas_hop = alphas.moveaxis(-1, 0)
hop = high_order_phase(phi_hop, alphas_hop)
start = time.perf_counter()
F_hop = hop.forward_matrix_prod(fr[None,])[0]
# F_hop = hop.adjoint_matrix_prod(F_hop[None,])[0]
torch.cuda.synchronize()
end = time.perf_counter()
F_hop = F_hop.cpu()
F_hop = normalize(F_hop, F_naive)
print(f'Elapsed time: {end - start}')

us, vs = svd_decomp_operator(hop, L=2, fast_axis='spatial')

# ------------------- Plot Both -------------------
if F_naive.ndim == 1:
    plt.figure(figsize=(14,7))
    plt.plot(F_naive.real, label='Naive', color='b')
    plt.plot(F_naive.imag, color='b', ls='--')
    plt.plot(F_hop.real, label='HOP', color='orange')
    plt.plot(F_hop.imag, color='orange', ls='--')
    plt.legend()
    
    plt.figure(figsize=(14,7))
    plt.plot((F_hop - F_naive).abs())
elif F_naive.ndim == 2:
    vmax = F_hop.abs().median() + 3 * F_hop.abs().std()
    plt.figure(figsize=(14,7))
    plt.subplot(221)
    plt.title('Naive')
    plt.imshow(F_naive.abs(), cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(F_naive.angle(), cmap='jet')
    plt.axis('off')
    plt.subplot(222)
    plt.title('HOP')
    plt.imshow(F_hop.abs(), cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(F_hop.angle(), cmap='jet')
    plt.axis('off')
    
    plt.figure(figsize=(14,7))
    plt.imshow((F_hop - F_naive).abs(), cmap='gray', vmin=0, vmax=vmax/5)
    
plt.show()