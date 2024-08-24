import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.utils import gen_grd
from mr_recon.imperfections.motion import motion_op
from mr_sim.coil_maps import surface_coil_maps
from mr_sim.phantoms import shepp_logan

# GPU
torch_dev = torch.device(5)

# Gen data
im_size = (220, 220)
phantom = shepp_logan(torch_dev).img(im_size)
pts = gen_grd(im_size).type(torch.float32).to(torch_dev)
pts = torch.cat([pts, pts[..., :1] * 0], dim=-1)
mps = surface_coil_maps(12, pts, phantom)
motion_params = -torch.tensor([
    0.1,
    0.0,
    0.0,
    0.0,
    0.0,
    -45/2,
], device=torch_dev)[None, :]

rm = motion_op(im_size).to(torch_dev)
img_rot = rm(phantom, motion_params)
mps_rot = rm(mps, motion_params)
 
plt.figure(figsize=(14, 7))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(mps_rot[i, 0].abs().cpu(), cmap='gray')
plt.show()

