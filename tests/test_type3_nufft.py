import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.utils import normalize
from mr_recon.linops import type3_nufft

# Set Seeds
torch.manual_seed(0)
np.random.seed(0)

# Random phis and alphas
torch_dev = torch.device(5)
# torch_dev = torch.device('cpu')
im_size = (32, 32,)
trj_size = (1000, 2,)
B = 4
N = 2
phis = torch.randn((B, *im_size), dtype=torch.float32, device=torch_dev) / 4
phis += torch.randn((B,), dtype=torch.float32, device=torch_dev)[:, None, None]
alphas = torch.randn((B, *trj_size), dtype=torch.float32, device=torch_dev) / 4
alphas += torch.randn((B,), dtype=torch.float32, device=torch_dev)[:, None, None]

# Create encoding matrix
phz = alphas.reshape((B, -1)).T @ phis.reshape((B, -1))
E = torch.exp(-2j * torch.pi * phz)
t3n = type3_nufft(phis, alphas, oversamp=2.0, width=6, use_toep=True)
x = torch.randn((N, *im_size), dtype=torch.complex64, device=torch_dev)
x_flt = x.reshape((N, -1)).T

# Debugging
kwargs = {'atol': 0.0, 'rtol': 0.05}
def plot_errs(s, s_gt):
    plt.figure()
    # plt.plot((s - s_gt).abs().flatten() / s_gt.abs().flatten())
    # plt.ylim(-kwargs['rtol']/2, kwargs['rtol']*5)
    plt.plot(s.abs().flatten() / s_gt.abs().flatten())
    plt.show()
    quit()

# Forward pass
y_gt = (E @ x_flt).T.reshape(N, *trj_size).cpu()
y = t3n.forward(x).cpu()
# y = normalize(y, y_gt)
if not torch.allclose(y, y_gt, **kwargs):
    print('Forward pass failed.')
    plot_errs(y, y_gt)
    
# Normal
x_nrm_gt = (E.H @ E @ x_flt).T.reshape((N, *im_size)).cpu()
x_nrm = t3n.normal(x).cpu()
# x_nrm = normalize(x_nrm, x_nrm_gt)
if not torch.allclose(x_nrm, x_nrm_gt, **kwargs):
    print('Normal pass failed.')
    plot_errs(x_nrm, x_nrm_gt)

# Adjoint
y_gt_flt = y_gt.reshape((N, -1)).T.to(torch_dev)
x_adj_gt = (E.conj().T @ y_gt_flt).T.reshape(N, *im_size).cpu()
x_adj = t3n.adjoint(y.to(torch_dev)).cpu()
# x_adj = normalize(x_adj, x_adj_gt)
if not torch.allclose(x_adj, x_adj_gt, **kwargs): 
    print('Adjoint pass failed.')
    plot_errs(x_adj, x_adj_gt)

print("All tests passed!")

