import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import time
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri

from mr_recon.algs import density_compensation
from mr_recon.linops import sense_linop, batching_params
from mr_recon.utils import np_to_torch, normalize
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import gridded_nufft, torchkb_nufft, sigpy_nufft, svd_nufft
from mr_sim.trj_lib import trj_lib

from igrog.grogify import imperfection_implicit_grogify, gridding_implicit_grogify, gridding_grogify
from igrog.training import training_params
from igrog.gridding import gridding_params

# Set seeds
np.random.seed(0)
torch.manual_seed(0)

# Params
figsize = (14, 7)
# im_size = (220, 220)
# z_slc = slice(None)
im_size = (220, 220, 220)
z_slc = 110
ninter = 16
ncoil = 6
R = 2
L = 12
lamda_l2 = 1e-3 * 0
max_iter = 10
max_eigen = 0.4
device_idx = 2
os_grid = 1.0
try: 
	torch_dev = torch.device(device_idx)
except:
	torch_dev = torch.device('cpu')

# Gen data
phantom = sp.shepp_logan(im_size)
# trj = mri.spiral(fov=1,
# 				 N=im_size[0],
# 				 f_sampling=.05,
# 				 R=1.0,
# 				 ninterleaves=ninter,
# 				 alpha=1.0,
# 				 gm=40e-3,
# 				 sm=100)
# trj = trj.reshape((-1, ninter, 2), order='F')[:, ::round(R), :]
mps = mri.birdcage_maps((ncoil, *im_size), r=1.0)
ros = slice(None)
grps = slice(None)
trs = slice(None)
trj = np.load('../../mr_data/threeT/data/trj.npy')[ros, grps, trs, :]
dcf = np.load('../../mr_data/threeT/data/dcf.npy')[ros, grps, trs]
# dcf = sp.to_device(mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx)))
dcf /= dcf.max()

# # Simulate ksp
# ksp = sp.nufft(phantom * mps, trj, oversamp=2.0, width=6)

# Move everything to torch
phantom = np_to_torch(phantom).to(torch_dev).type(torch.complex64)
trj = np_to_torch(trj).to(torch_dev).type(torch.float32)
mps = np_to_torch(mps).to(torch_dev).type(torch.complex64)
dcf = np_to_torch(dcf).to(torch_dev).type(torch.float32)
# ksp = np_to_torch(ksp).to(torch_dev).type(torch.complex64)

# Make nufft linops
bparams = batching_params(coil_batch_size=1)
# nufft_nc = torchkb_nufft(im_size=im_size, torch_dev)
nufft_sim = sigpy_nufft(im_size=im_size, width=6, oversamp=2.0)
A_sim = sense_linop(im_size=im_size,
					trj=trj,
					mps=mps,
					dcf=dcf,
					bparams=bparams,
					nufft=nufft_sim,
					use_toeplitz=False,)

# Simulate with big nufft
ksp = A_sim(phantom)
phantom = phantom.cpu().numpy()
print(f'Finished simulating data')

# KB Recon
nufft_nc = sigpy_nufft(im_size=im_size, width=4, oversamp=1.25)
A_nc = sense_linop(im_size=im_size,
					trj=trj,
					mps=mps,
					dcf=dcf,
					bparams=bparams,
					nufft=nufft_nc,
					use_toeplitz=False,)


# NUFFT Recon
start = time.perf_counter()
img_nufft = CG_SENSE_recon(A=A_nc, ksp=ksp,
							lamda_l2=lamda_l2,
							max_iter=max_iter,
							max_eigen=max_eigen)
torch.cuda.synchronize()
end = time.perf_counter()
img_nufft = normalize(img_nufft.cpu().numpy(), phantom)
nrmse_nufft = 100 * np.linalg.norm(img_nufft - phantom) / np.linalg.norm(phantom)
t_nufft = (end - start)

start = time.perf_counter()
img_nufft = CG_SENSE_recon(A=A_nc, ksp=ksp,
						lamda_l2=lamda_l2,
						max_iter=max_iter,
						max_eigen=max_eigen)
torch.cuda.synchronize()
end = time.perf_counter()
img_nufft = normalize(img_nufft.cpu().numpy(), phantom)
nrmse_nufft = 100 * np.linalg.norm(img_nufft - phantom) / np.linalg.norm(phantom)
t_nufft = (end - start)

# SVD NUFFT Recons
imgs_imperf = []
rmses = []
Ls = []
ts = []
plt.figure(figsize=figsize)
nufft = svd_nufft(im_size, os_grid, L, n_batch_size=1)
A = sense_linop(im_size=im_size, trj=trj, mps=mps, dcf=dcf,
				nufft=nufft,
				use_toeplitz=False,
				bparams=bparams)
L_start = 1
for l_iter in range(L_start, L):
	nufft.n_svd = l_iter
	
	print(f'\nRunning L = {l_iter}')
	start = time.perf_counter()
	img = CG_SENSE_recon(A=A, ksp=np_to_torch(ksp).to(torch_dev), 
							lamda_l2=lamda_l2,
							max_iter=max_iter,
							max_eigen=max_eigen)
	torch.cuda.synchronize()
	end = time.perf_counter()
	t_imperf = (end - start)

	img = normalize(img.cpu().numpy(), phantom)
	imgs_imperf.append(img)
	nrmse = 100 * np.linalg.norm(img - phantom) / np.linalg.norm(phantom)
	rmses.append(nrmse)
	Ls.append(l_iter)
	ts.append(t_imperf)

	plt.subplot(3, 4, l_iter)
	plt.title(f'L = {l_iter} NRMSE = {nrmse:.2f} T = {t_imperf:.2f} s')
	img = np.abs(img)
	vmax = np.median(img) + 3 * np.std(img)
	plt.imshow(img[z_slc], cmap='gray', vmin=0, vmax=vmax)
	plt.axis('off')
plt.tight_layout()

plt.figure(figsize=figsize)
plt.plot(Ls, rmses)
plt.xlabel('L')
plt.ylabel('% NRMSE')
plt.axhline(nrmse_nufft, color='r', linestyle='--')

plt.figure(figsize=figsize)
plt.scatter(ts, rmses, color='blue', label='SVD')
plt.scatter([t_nufft], [nrmse_nufft], color='red', label='NUFFT')
plt.xlabel('Time (s)')
plt.ylabel('% NRMSE')
plt.legend()

plt.figure(figsize=figsize)
plt.title(f'Nufft Recon T = {t_nufft:.2f} s')
img = np.abs(img_nufft)
vmax = np.median(img) + 3 * np.std(img)
plt.imshow(img[z_slc], cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.tight_layout()

plt.figure(figsize=figsize)
plt.title(f'SVD_nufft Recon T = {t_imperf:.2f} s (L = {L})')
img = np.abs(imgs_imperf[-1])
vmax = np.median(img) + 3 * np.std(img)
plt.imshow(img[z_slc], cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.tight_layout()
plt.show()
quit()
					