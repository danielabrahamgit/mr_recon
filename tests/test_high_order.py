import time
import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import gen_grd, np_to_torch, torch_to_np
from mr_recon.spatial import spatial_resize
from mr_recon.imperfections.coco import coco_imperfection
from mr_recon.imperfections.spatio_temporal_imperf import high_order_phase, alphas_phis_from_B0, alphas_phis_from_coco
from mr_recon.imperfections.imperf_decomp import temporal_segmentation
from mr_recon.linops import type3_nufft

from tqdm import tqdm
from einops import einsum, rearrange

# Set seeds
np.random.seed(0)
torch.manual_seed(0)

# GPU stuff
torch_dev = torch.device(0)
# torch_dev = torch.device('cpu')
    
# Recon Stuff
im_size = (192,)*3
ro_time = '11ms'
# ro_time = '3ms'
use_toeplitz = False
max_iter = 5
lamda_l2 = 1e-2 
dt = 5e-6
R = 1

# Load trajectory + B0 Data for a 3D lowfield MRF Sequence
pth = f'/local_mount/space/tiger/1/users/abrahamd/mr_data/mrf_coco/data_{ro_time}/'
trj = np_to_torch(np.load(pth + 'trj.npy')).to(torch_dev).type(torch.float32)
if ro_time == '11ms':
    b0 = np_to_torch(np.load(pth + 'b0.npy')).to(torch_dev).type(torch.float32)
print(f'trj size = {trj.shape}')

# Create phis and alphas
xyz = gen_grd(im_size, (0.22,)*3).to(torch_dev)
a1, p1 = alphas_phis_from_coco(trj, xyz, (0.22,)*3, dt, 0.55)
a2, p2 = alphas_phis_from_B0(b0, trj.shape[:-1], dt)
a2 = a2.expand((1, *a1.shape[1:]))
alphas = torch.cat([a1, a2], dim=0)
phis = torch.cat([p1, p2], dim=0)

# Subsample to smaller problem
ros = slice(None)
trs = slice(None)
groups = slice(None)
ros = slice(None, None, 10)
# groups = slice(2, 3)
# trs = slice(3, 4) 
alphas = alphas[:, ros, groups, trs].reshape((alphas.shape[0], -1))
# alphas_flt = alphas.reshape((alphas.shape[0], -1))
# rnd_inds = torch.randperm(alphas_flt.shape[1])[9_000:10_000]
# alphas = alphas_flt[:, rnd_inds]
# alphas = alphas.reshape((alphas.shape[0], -1))

from mr_recon.imperfections.field import alpha_phi_svd, alpha_segementation
phis = spatial_resize(phis, (64,)*3, method='fourier').real
# alpha_phi_svd(phis, alphas, L=10, L_batch_size=10, num_iter=15//5)
alpha_segementation(phis[:4], alphas[:4], L=10, L_batch_size=10, interp_type='lstsq')
