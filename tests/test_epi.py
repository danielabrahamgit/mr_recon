import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import np_to_torch, torch_to_np
from mr_recon.linops import subspace_linop, sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import gridded_nufft
from mr_recon.imperfections.field import field_handler
from mr_recon.imperfections.main_field import main_field_imperfection

# Set seed 
np.random.seed(1230)
torch.manual_seed(1230)

# GPU
device_idx = 6
device = torch.device(device_idx)

# Load data
slc = 0
ksp = np.load('./data/ksp.npy')[..., slc][..., :-1]
trj = np.load('./data/trj.npy')[..., :-1, :]
mps = np.load('./data/mps.npy')[..., slc]
phi = np.load('./data/phi.npy').astype(np.complex64)[..., :-1]
te = np.load('./data/te.npy')[..., :-1]
b0 = np.load('./data/b0.npy')[..., slc]

# Print shapes 
print(f'ksp shape = {ksp.shape}')
print(f'trj shape = {trj.shape}')
print(f'mps shape = {mps.shape}')
print(f'phi shape = {phi.shape}')
print(f'te shape = {te.shape}')
print(f'b0 shape = {b0.shape}')

field_obj = main_field_imperfection(b0_map=np_to_torch(b0).to(device).type(torch.float32),
                                    trj_dims=(1, 1, ksp.shape[-1]),
                                    dt=te[1] - te[0],
                                    L=15,
                                    method='svd',
                                    interp_type='lstsq',)

# # Custom alpha phis since weird time evolution
# nro = trj.shape[0]
# ro_lin = 0e-3 * torch.arange(-(nro//2), nro//2, device=device) / nro
# tup = (slice(None),) + (None,) * (trj.ndim - 1)
# # alphas = ro_lin[tup] + np_to_torch(te[None, None, :, None]).to(device).type(torch.float32)
# alphas = np_to_torch(te[None, None, :, None]).to(device).type(torch.float32)
# phis = -np_to_torch(b0)[None, ...].to(device).type(torch.float32)
# field_obj = field_handler(alphas, phis, 
#                           nseg=15,
#                           method='svd',
#                           interp_type='lstsq',
#                           quant_type='uniform')
# field_obj._plot_approx_err(t_slice=(0, 0, slice(None)))

# Make linop
# nufft = None
nufft = gridded_nufft(mps.shape[1:], device_idx)
bparams = batching_params(
    coil_batch_size=2,
    sub_batch_size=3,
    field_batch_size=4
)
A = subspace_linop(im_size=mps.shape[1:],
                   trj=np_to_torch(trj).to(device),
                   mps=np_to_torch(mps).to(device),
                   phi=np_to_torch(phi).to(device),
                   field_obj=field_obj,
                   bparams=bparams,
                   nufft=nufft)

# Run recon
coeffs = CG_SENSE_recon(A=A,
                        ksp=np_to_torch(ksp).to(device),
                        max_iter=15,
                        max_eigen=1.0,
                        lamda_l2=0.0).cpu()

# Show results
plt.figure(figsize=(14,7))
plt.suptitle(f'Nseg = {field_obj.nseg}, Interp = {field_obj.interp_type}')
for i in range(coeffs.shape[0]):
    plt.subplot(1, coeffs.shape[0], i+1)
    plt.title(f'Subspace Coeff {i + 1}')
    plt.imshow(coeffs[i].abs(), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()