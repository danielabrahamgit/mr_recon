import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import time
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri

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
im_size = (220, 220)
ninter = 16
ncoil = 32
R = 4
L = 14
lamda_l2 = 1e-3 * 0
max_iter = 500
max_eigen = 0.4
device_idx = 5
os_grid = 1.2
try: 
	torch_dev = torch.device(device_idx)
except:
	torch_dev = torch.device('cpu')

# Gen data
phantom = sp.shepp_logan(im_size)
trj = mri.spiral(fov=1,
				 N=220,
				 f_sampling=.05,
				 R=1.0,
				 ninterleaves=ninter,
				 alpha=1.0,
				 gm=40e-3,
				 sm=100)
trj = trj.reshape((-1, ninter, 2), order='F')[:, ::round(R), :]
# trj_lb = trj_lib(im_size)
# trj = trj_lb.gen_trj_2d('spi', 
#                         n_read=im_size[0] * 4,
#                         n_shots=ninter,
#                         R=R)
mps = mri.birdcage_maps((ncoil, *im_size), r=1.0)
dcf = sp.to_device(mri.pipe_menon_dcf(trj, im_size, device=sp.Device(device_idx)))
dcf /= dcf.max()

# Simulate with sigpy 
ksp = sp.nufft(phantom * mps, trj, oversamp=2.0, width=6)

# Move everything to torch
trj = np_to_torch(trj).to(torch_dev).type(torch.float32)
mps = np_to_torch(mps).to(torch_dev).type(torch.complex64)
dcf = np_to_torch(dcf).to(torch_dev).type(torch.float32)
ksp = np_to_torch(ksp).to(torch_dev).type(torch.complex64)

# Make nufft linops
bparams = batching_params(coil_batch_size=ncoil)
# nufft_nc = torchkb_nufft(im_size=im_size, torch_dev)
nufft_nc = sigpy_nufft(im_size=im_size)
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
nrmse_nufft = np.linalg.norm(img_nufft - phantom) / np.linalg.norm(phantom)
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
nufft = svd_nufft(im_size, os_grid, L)
A = sense_linop(im_size=im_size, trj=trj, mps=mps, dcf=dcf,
				nufft=nufft,
				use_toeplitz=False,
				bparams=bparams)
for l_iter in range(1, L):
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

	plt.subplot(3, 5, l_iter)
	plt.title(f'L = {l_iter} NRMSE = {nrmse:.2f} T = {t_imperf:.2f} s')
	img = np.abs(img)
	vmax = np.median(img) + 3 * np.std(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=vmax)
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
plt.imshow(img, cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.tight_layout()
plt.show()
quit()
					