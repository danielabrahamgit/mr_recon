import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.linops import subspace_linop, subspace_linop_old, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import np_to_torch, torch_to_np, normalize

# Load MRF data
import sys
sys.path.append('/local_mount/space/tiger/1/users/abrahamd/')
from mr_data.data_loader import data_loader
filename = 'mrf_2d'
dl = data_loader(filename)
img_consts, data_dict = dl.load_dataset()
mps = data_dict['mps']
phi = data_dict['phi']
trj = data_dict['trj']
dcf = data_dict['dcf']
ksp = data_dict['ksp']
ksp_cal = data_dict['ksp_cal']
ksp /= np.abs(ksp).max()
ksp_cal /= np.abs(ksp_cal).max()
dcf /= dcf.max()

# Consts
im_size = mps.shape[1:]
max_iter = 5
max_eigen = 1.0

bparams = batching_params(
    coil_batch_size=2,
    sub_batch_size=3
)

# Make linop
torch_dev = torch.device(4)
A = subspace_linop(im_size, 
                   trj=np_to_torch(trj).to(torch_dev),
                   mps=np_to_torch(mps).to(torch_dev),
                   dcf=np_to_torch(dcf).to(torch_dev),
                   phi=np_to_torch(phi).to(torch_dev),
                   use_toeplitz=False,
                   bparams=bparams)
A_toep = subspace_linop(im_size, 
                   trj=np_to_torch(trj).to(torch_dev),
                   mps=np_to_torch(mps).to(torch_dev),
                   dcf=np_to_torch(dcf).to(torch_dev),
                   phi=np_to_torch(phi).to(torch_dev),
                   use_toeplitz=True,
                   bparams=bparams)

# Recon with both
ksp = np_to_torch(ksp).to(torch_dev)
x = CG_SENSE_recon(A, ksp, max_eigen=max_eigen, max_iter=max_iter).cpu()
x_toep = CG_SENSE_recon(A_toep, ksp, max_eigen=max_eigen, max_iter=max_iter).cpu()

# Normalize
x_toep = np_to_torch(normalize(x_toep, x))

# Compare
def plot(coeffs, title=''):
    plt.figure(figsize=(14, 7))
    plt.suptitle(title)
    vmax_first = None
    for i in range(coeffs.shape[0]):
        plt.subplot(1, coeffs.shape[0], i + 1)
        img = torch.abs(coeffs)[i]
        img = torch.rot90(img, 1, [0, 1])
        vmin = 0.0
        vmax = (torch.median(img) + 3 * torch.std(img)).item()
        if vmax_first is None:
            vmax_first = vmax
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f'Coeff {i + 1} ({round(vmax_first / vmax)}X)')
        plt.axis('off')
    plt.tight_layout()
plot(x, 'Standard')
plot(x_toep, 'Toeplitz')
plot(x - x_toep, 'Diff')
plt.show()
