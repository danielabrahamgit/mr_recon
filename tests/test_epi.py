import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import np_to_torch, torch_to_np
from mr_recon.linops import subspace_linop, sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon, FISTA_recon
from mr_recon.prox import L1Wav
from mr_recon.fourier import gridded_nufft
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

imperf_model = main_field_imperfection(b0_map=-np_to_torch(b0).to(device).type(torch.float32),
                                    trj_dims=(1, 1, ksp.shape[-1]),
                                    dt=te[1] - te[0],
                                    L=30,
                                    method='svd',
                                    interp_type='zero',)

# Make linop
# nufft = None
nufft = gridded_nufft(mps.shape[1:], device_idx)
bparams = batching_params(
    coil_batch_size=mps.shape[0],
    sub_batch_size=3,
    field_batch_size=4
)
A = subspace_linop(im_size=mps.shape[1:],
                   trj=np_to_torch(trj).to(device),
                   mps=np_to_torch(mps).to(device),
                   phi=np_to_torch(phi).to(device),
                   imperf_model=imperf_model,
                   bparams=bparams,
                   nufft=nufft)

# Run recon
proxg = L1Wav((phi.shape[0], *mps.shape[1:]), 
              lamda=5e0, 
              axes=(-2, -1)).forward
coeffs = FISTA_recon(A, np_to_torch(ksp).to(device), proxg, max_iter=40, verbose=True).cpu()
# np.save('./data/coeffs.npy', coeffs.numpy())

# Show results
plt.figure(figsize=(14,7))
plt.suptitle(f'Nseg = {imperf_model.L}, Interp = {imperf_model.interp_type}')
for i in range(coeffs.shape[0]):
    plt.subplot(1, coeffs.shape[0], i+1)
    plt.title(f'Subspace Coeff {i + 1}')
    plt.imshow(coeffs[i].abs(), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()