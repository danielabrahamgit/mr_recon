import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from scipy.ndimage import rotate
from mr_recon.imperfections.motion import motion_op
from mr_recon.imperfections.motion_op_torch import rigid_motion

# Gen data
im_size = (220, 220)
phantom = torch.from_numpy(sp.shepp_logan(im_size))

ang = -torch.pi/8
motion_params = torch.tensor([
    220/10,
    0.0,
    0.0,
    0.0,
    0.0,
    ang,
])[None, :]

import time
start = time.time()
rm = rigid_motion(ishape=im_size)
rm._reset_motion_params(motion_params)
img = rm._apply(phantom)
img = img[:, :, :, 0, 0]
print(time.time() - start)

motion_params = -torch.tensor([
    0.1,
    0.0,
    0.0,
    0.0,
    0.0,
    -45/2,
])[None, :]

import time
start = time.time()
rm = motion_op(im_size)
img_mine = rm(phantom, motion_params)
print(time.time() - start)


plt.figure()
plt.imshow(img.abs()[0], vmin=0, vmax=1)
plt.figure()
plt.imshow(img_mine.abs()[0], vmin=0, vmax=1)
plt.show()

