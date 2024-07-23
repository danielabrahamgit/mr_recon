import time
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.calib import calc_coil_subspace
from mr_recon.utils import np_to_torch, normalize
from mr_recon.multi_coil.coil_est import csm_from_grappa, csm_from_espirit

# Sim params
nc = 8
ndim = 2
im_size = (200,) * ndim
cal_size = 32
device_idx = 5
try:
    torch_dev = torch.device(device_idx)
except:
    torch_dev = torch.device('cpu')

# Estimation params
num_kerns = 500
num_src = 12
crp = 0.0
max_iter = 100

# Synth data
mps = mri.birdcage_maps((nc, *im_size), r=1.0).astype(np.complex64)
phantom = sp.shepp_logan(im_size)
ksp_cal = sp.fft(mps * phantom, axes=[-2, -1])
ksp_cal = sp.resize(ksp_cal, (nc, cal_size, cal_size))

# Move to torch
img_cal = np_to_torch(mps * phantom).to(torch_dev).type(torch.complex64)
ksp_cal = np_to_torch(ksp_cal).to(torch_dev).type(torch.complex64)
mps = np_to_torch(mps).to(torch_dev).type(torch.complex64)

# Compress
coil_sub, img_cal, ksp_cal, mps = calc_coil_subspace(ksp_cal, 0.95, img_cal, ksp_cal, mps)

# Estimate coil maps with GRAPPA
start = time.perf_counter()
mps_grap, evals_grap = csm_from_grappa(ksp_cal, im_size,
                                       num_kerns=num_kerns, 
                                       num_src=num_src, 
                                       kernel_width=5, 
                                       crp=1e9, 
                                       max_iter=max_iter)
end = time.perf_counter()
grap_time = end - start

# Estimate coil maps with ESPIRiT
start = time.perf_counter()
mps_esp, evals_esp = csm_from_espirit(ksp_cal, im_size, 
                                      kernel_width=5,
                                      thresh=0.02,
                                      crp=crp, 
                                      max_iter=max_iter)
end = time.perf_counter()
esp_time = end - start

# Normalize maps
mps = mps.cpu().numpy()
phz = np.exp(1j * np.angle(mps))
mps_grap = normalize(mps_grap.cpu().numpy() * phz, mps, ofs=False, mag=False)
mps_esp = normalize(mps_esp.cpu().numpy() * phz, mps, ofs=False, mag=False)
evals_grap = evals_grap.cpu().numpy()
evals_esp = evals_esp.cpu().numpy()
msk = np.abs(phantom) > 0   

# Plots
titles = ['GT', 'GRAPPA', 'ESPIRiT']
times = [0.0, grap_time, esp_time]
est_mps = [mps, mps_grap, mps_esp]
est_evals = [np.abs(phantom), evals_grap, evals_esp]
msk_img = msk * 0 + 1
for c in range(3):
    fig, axs = plt.subplots(3, 3, figsize=(8.75,9))
    plt.suptitle(f'Coil {c+1}')
    for i in range(3):
        mp = est_mps[i][c]
        ev = est_evals[i]

        # Show maps
        vmax = np.abs(mp * msk).max()
        # axs[0,i].set_title(f'Coil maps {titles[i]}\nTime = {times[i]:.2f}s')
        axs[0,i].set_title(f'Coil maps {titles[i]}')
        # axs[0,i].set_title(f'Coil maps {titles[i]}')
        axs[0,i].imshow(np.abs(mp) * msk_img, cmap='gray', vmin=0, vmax=vmax)
        axs[0,i].axis('off')

        # Show Phase
        axs[1,i].imshow(np.angle(mp), cmap='jet', vmin=-np.pi, vmax=np.pi)
        axs[1,i].axis('off')

        # Show eigenvalues
        axs[2,i].imshow(ev, cmap='gray')
        axs[2,i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
plt.show()