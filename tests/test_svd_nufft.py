import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

import os
import time
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri

from mr_recon.linops import sense_linop, batching_params
from mr_recon.utils import np_to_torch, normalize
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import gridded_nufft, torchkb_nufft, sigpy_nufft
from mr_recon.imperfections.off_grid import off_grid_imperfection
from mr_sim.trj_lib import trj_lib

from igrog.grogify import imperfection_implicit_grogify, gridding_implicit_grogify, gridding_grogify
from igrog.training import training_params
from igrog.gridding import gridding_params

# Set seeds
np.random.seed(0)
torch.manual_seed(0)

# Params
im_size = (220, 220)
ninter = 16
ncoil = 32
R = 4
lamda_l2 = 1e-3 * 0
max_iter = 100
device_idx = -1
try: 
	torch_dev = torch.device(device_idx)
except:
	torch_dev = torch.device('cpu')

def loss(x, y):
	err = x - y
	return torch.mean(torch.abs(err) ** 2)

grid_params_grog = gridding_params(num_inputs=1,
								   kern_width=1.0,
								   interp_readout=True,
								   grid=True)
grid_params_igrog = gridding_params(num_inputs=5,
									kern_width=2.0,
									interp_readout=True,
									# oversamp_grid=2.0,
									grid=False)
train_params = training_params(epochs=30,
							   batch_size=2**10,
							   batches_per_epoch=100,
							  #  loss=loss,
							   l2_reg=1e-7*0,)

# Gen data
phantom = sp.shepp_logan(im_size)
trj = mri.spiral(fov=1,
				 N=220,
				 f_sampling=.25,
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

plot = True
if plot == False:

	# Make imperfection model
	trj_grd = torch.round(trj * grid_params_igrog.oversamp_grid) / grid_params_igrog.oversamp_grid
	grid_deviations = trj - trj_grd
	L = 14
	imperf_model = off_grid_imperfection(im_size=im_size,
											grid_deviations=grid_deviations,
											L=L,
											method='svd',
											interp_type='zero',
											verbose=True,)
	spatial_funcs = imperf_model.spatial_funcs.clone()
	temporal_funcs = imperf_model.temporal_funcs.clone()
	
	# Make linops
	bparams = batching_params(coil_batch_size=ncoil, field_batch_size=L)
	nufft = gridded_nufft(im_size=im_size, device_idx=device_idx, grid_oversamp=grid_params_igrog.oversamp_grid)
	# nufft_nc = torchkb_nufft(im_size=im_size, device_idx=device_idx)
	nufft_nc = sigpy_nufft(im_size=im_size, device_idx=device_idx)
	A_nc = sense_linop(im_size=im_size,
						trj=trj,
						mps=mps,
						dcf=dcf,
						bparams=bparams,
						nufft=nufft_nc,
						use_toeplitz=False,)
	A = sense_linop(im_size=im_size,
					trj=trj_grd,
					mps=mps,
					dcf=dcf,
					bparams=bparams,
					nufft=nufft,
					imperf_model=imperf_model,
					use_toeplitz=False,)
	nruns = 2
	for run in range(nruns):

		# NUFFT Recon
		start = time.perf_counter()
		img_nufft = CG_SENSE_recon(A=A_nc, ksp=np_to_torch(ksp).to(torch_dev),
								   lamda_l2=lamda_l2,
								   max_iter=max_iter)
		torch.cuda.synchronize()
		end = time.perf_counter()
		img_nufft = normalize(img_nufft.cpu().numpy(), phantom)
		nrmse_nufft = np.linalg.norm(img_nufft - phantom) / np.linalg.norm(phantom)
		t_nufft = (end - start)
		if run == 0:
			start = time.perf_counter()
			img_nufft = CG_SENSE_recon(A=A_nc, ksp=np_to_torch(ksp).to(torch_dev),
									lamda_l2=lamda_l2,
									max_iter=max_iter)
			torch.cuda.synchronize()
			end = time.perf_counter()
			img_nufft = normalize(img_nufft.cpu().numpy(), phantom)
			nrmse_nufft = np.linalg.norm(img_nufft - phantom) / np.linalg.norm(phantom)
			t_nufft = (end - start)

		# Imperfect Recons
		imgs_imperf = []
		rmses = []
		Ls = []
		ts = []
		for l_iter in range(1, L):
			imperf_model.L = l_iter
			imperf_model.spatial_funcs = spatial_funcs[:l_iter]
			imperf_model.temporal_funcs = temporal_funcs[:l_iter]
			A.imperf_rank = l_iter
			A.bparams.field_batch_size = l_iter
			
			start = time.perf_counter()
			img = CG_SENSE_recon(A=A, ksp=np_to_torch(ksp).to(torch_dev), 
								 lamda_l2=lamda_l2,
								 max_iter=max_iter)
			torch.cuda.synchronize()
			end = time.perf_counter()
			t_imperf = (end - start)

			img = normalize(img.cpu().numpy(), phantom)
			imgs_imperf.append(img)
			nrmse = np.linalg.norm(img - phantom) / np.linalg.norm(phantom)
			rmses.append(nrmse)
			Ls.append(imperf_model.L)
			ts.append(t_imperf)
			
		# Save data
		imgs_imperf = np.array(imgs_imperf)
		rmses = np.array(rmses)
		Ls = np.array(Ls)
		ts = np.array(ts)
		if device_idx == -1:
			filename = f'./data/svd_nufft/sweep_data_cpu_{run}.npz'
		else:
			filename = f'./data/svd_nufft/sweep_data_gpu_{run}.npz'
		np.savez(filename, 
						#  imgs_imperf=imgs_imperf, 
						 rmses=rmses, 
						 Ls=Ls, 
						 ts=ts, 
						#  img_nufft=img_nufft, 
						 nrmse_nufft=nrmse_nufft, 
						 t_nufft=t_nufft)
else:
	Ls_all = None
	rmses_all = None
	ts_all = None
	img_nufft_all = None
	nrmse_nufft_all = None
	t_nufft_all = None

	if device_idx == -1:
		cpu_gpu = 'cpu'
	else:
		cpu_gpu = 'gpu'
	for filename in os.listdir('./data/svd_nufft/'):
		if filename.endswith(".npz") and cpu_gpu in filename:
				data = np.load('./data/svd_nufft/' + filename)
				Ls = data['Ls']
				rmses = data['rmses']
				ts = data['ts']
				# img_nufft = data['img_nufft']
				nrmse_nufft = data['nrmse_nufft'][None,]
				t_nufft = data['t_nufft'][None,]

				if Ls_all is None:
					Ls_all = Ls[None,]
					rmses_all = rmses[None,]
					ts_all = ts[None,]
					# img_nufft_all = img_nufft[None,]
					nrmse_nufft_all = np.array(nrmse_nufft)[None,]
					t_nufft_all = np.array(t_nufft)[None,]
				else:
					Ls_all = np.concatenate((Ls_all, Ls[None,]), axis=0)
					rmses_all = np.concatenate((rmses_all, rmses[None,]), axis=0)
					ts_all = np.concatenate((ts_all, ts[None,]), axis=0)
					# img_nufft_all = np.concatenate((img_nufft_all, img_nufft[None,]), axis=0)
					nrmse_nufft_all = np.concatenate((nrmse_nufft_all, np.array(nrmse_nufft)[None,]), axis=0)
					t_nufft_all = np.concatenate((t_nufft_all, np.array(t_nufft)[None,]), axis=0)
	
	# plt.figure()
	# plt.title(f'L = {imperf_model.L} NRMSE = {nrmse:.2e} T = {t_imperf:.2f} s')
	# img = np.abs(img)
	# vmax = np.median(img) + 3 * np.std(img)
	# plt.imshow(img, cmap='gray', vmin=0, vmax=vmax)
	# plt.axis('off')
	# plt.tight_layout()

	# plt.figure()
	# plt.title(f'Nufft Recon T = {t_nufft:.2f} s')
	# img = np.abs(img_nufft)
	# vmax = np.median(img) + 3 * np.std(img)
	# plt.imshow(img, cmap='gray', vmin=0, vmax=vmax)
	# plt.axis('off')
	# plt.tight_layout()
					
	# plt.imshow(ts_all)
	# plt.show()
	# quit()
					
	font = {'family' : 'monospace',
					'weight' : 'bold'}
	matplotlib.rc('font', **font)
	matplotlib.rc('lines', lw=3)

	plt.figure(figsize=(14, 7))
	rmses_nufft_mean = np.mean(nrmse_nufft_all, axis=0) * np.ones(Ls_all.shape[1])
	rmses_nufft_std = np.std(nrmse_nufft_all, axis=0) * np.ones(Ls_all.shape[1])
	rmses_mean = np.mean(rmses_all, axis=0)
	rmses_std = np.std(rmses_all, axis=0)
	plt.errorbar(Ls_all[0], rmses_mean, yerr=rmses_std)
	plt.xlabel('L')
	plt.ylabel('% NRMSE')
	plt.errorbar(Ls_all[0], rmses_nufft_mean, rmses_nufft_std, color='r')
	plt.tight_layout()

	plt.figure(figsize=(14, 7))
	ts_nufft_mean = np.mean(t_nufft_all, axis=0) * np.ones(Ls_all.shape[1])
	ts_nufft_std = np.std(t_nufft_all, axis=0) * np.ones(Ls_all.shape[1])
	ts_mean = np.mean(ts_all, axis=0)
	ts_std = np.std(ts_all, axis=0)
	plt.errorbar(Ls_all[0], ts_mean, yerr=ts_std)
	plt.xlabel('L')
	plt.ylabel('Wall Time [s]')
	plt.errorbar(Ls_all[0], ts_nufft_mean, ts_nufft_std, color='r')
	plt.tight_layout()

	plt.show()
