import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from pyeyes import ComparativeViewer

from mr_recon.utils import gen_grd, normalize, rotation_matrix, quantize_data
from mr_recon.linops import type3_nufft
from mr_recon.spatial import kernel_interp, spatial_resize_poly
from mr_recon.imperfections.field import coco_to_phis_alphas, rescale_phis_alphas, alpha_phi_svd
from mr_sim.grad_utils import design_spiral_trj, design_epi_trj

from einops import einsum

# Set Seeds
torch.manual_seed(0)
np.random.seed(0)

# Paramss
torch_dev = torch.device(5)
# torch_dev = torch.device('cpu')
im_size = (300, 300)
solve_size = (50, 50)
fov = 0.22 # m

# Sim oblique spiral
# spi = design_spiral_trj(100e-3, 500, res=fov/im_size[0], fov=fov, nshots=1)[:, 0, :]
# torch.save(spi, 'spi.pt')
# quit()
spi = torch.load('spi.pt', weights_only=True).type(torch.float32)
spi = spi[::10]
crds = gen_grd(im_size, (fov,)*2)
spi = torch.cat([spi, spi[:, :1]*0], dim=1)
crds = torch.cat([crds, crds[..., :1]*0], dim=-1)
rot = rotation_matrix(torch.tensor([1.0, 0.0, 0.0]), torch.ones(1)*np.pi/2)[0]
spi = spi @ rot
crds = crds @ rot

# Coco field coeffs
phis, alphas = coco_to_phis_alphas(spi.to(torch_dev), crds.to(torch_dev), 3, 0, dt=10*2e-6)
phis = phis[:2]
alphas = alphas[:2]
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
phis = phis_nrm
alphas = alphas_nrm
B = phis.shape[0]

# Make phi smaller
phis = spatial_resize_poly(phis, solve_size, order=3)

# Subsample alphas via kmeans clustering
K = 100
betas, idxs = quantize_data(alphas.moveaxis(0, -1), K, method='cluster')
betas = betas.moveaxis(-1, 0)

# Solve smaller svd problem
svd_params = {'L': 15, 'use_type3': False}
bsmall, hsmall = alpha_phi_svd(phis, betas, **svd_params)

# Interpolate alphas via rbf
hbig = kernel_interp(betas.T, hsmall.T, alphas.T, kern_param=.4)
hbig = hbig.T

# SVD regular 
b, h = alpha_phi_svd(phis, alphas, **svd_params)

# Fix midpoint prob
phz_0 = torch.exp(-2j * np.pi * (phis_mp @ alphas_mp))
phz_spat = torch.exp(-2j * torch.pi * einsum(phis, alphas_mp, 'B ... , B -> ...')) * phz_0
phz_temp = torch.exp(-2j * np.pi * einsum(phis_mp, alphas, 'B, B ... -> ...'))
b *= phz_spat
h *= phz_temp
bsmall *= phz_spat
hbig *= phz_temp
phis = phis + phis_mp[:, None, None]
alphas = alphas + alphas_mp[:, None]

# Pyeyes to compare both
phz_gt = torch.exp(-2j * np.pi * einsum(phis, alphas, 'B ..., B T -> ... T'))
phz = einsum(b, h, 'L ..., L T -> ... T')
phz_aprox = einsum(bsmall, hbig, 'L ..., L T -> ... T')
cv = ComparativeViewer({'phz_gt': phz_gt.cpu(), 'bh': phz.cpu(), 'bh_aprox': phz_aprox.cpu()},
                       ['X', 'Y', 'T'],
                       ['Y', 'X'])
cv.launch()