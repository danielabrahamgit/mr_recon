import time
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mri

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.utils import np_to_torch, torch_to_np, normalize
from mr_recon.multi_coil.coil_est import csm_from_kernels, csm_from_espirit
from mr_recon.multi_coil.grappa_est import gen_source_vectors_rot, train_kernels

# Sim params
nc = 8
ndim = 2
im_size = (200,) * ndim
cal_size = 32
device_idx = -1
try:
    torch_dev = torch.device(device_idx)
except:
    torch_dev = torch.device('cpu')

# Estimation params
num_kerns = 1
num_inputs = 5
line_width = 6
ofs = 0.15
lamda_tikonov = 5e-3
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

# Estimate coil maps with GRAPPA
src_vecs = gen_source_vectors_rot(num_kerns=num_kerns, num_inputs=num_inputs, ndim=ndim, ofs=ofs, line_width=line_width)
start = time.perf_counter()
src_vecs = src_vecs.to(torch_dev).type(torch.float32)
kerns = train_kernels(img_cal, src_vecs, fast_method=True, lamda_tikonov=lamda_tikonov)
mps_grap, evals_grap = csm_from_kernels(kerns, src_vecs, im_size, crp=crp, max_iter=max_iter)
end = time.perf_counter()
grap_time = end - start

# Estimate coil maps with ESPIRiT
start = time.perf_counter()
mps_esp, evals_esp = csm_from_espirit(ksp_cal, im_size, crp=crp, max_iter=max_iter)
end = time.perf_counter()
esp_time = end - start

# Normalize maps
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
for c in range(nc):
    fig, axs = plt.subplots(3, 3, figsize=(8.75,9))
    plt.suptitle(f'Coil {c+1}')
    for i in range(3):
        mp = est_mps[i][c]
        ev = est_evals[i]

        # Show maps
        vmax = np.abs(mps[c] * msk).max()
        # axs[0,i].set_title(f'Coil maps {titles[i]}\nTime = {times[i]:.2f}s')
        axs[0,i].set_title(f'Coil maps {titles[i]}')
        axs[0,i].imshow(np.abs(mp) * msk_img, cmap='gray', vmin=0, vmax=vmax)
        axs[0,i].axis('off')

        # Show errors
        err = np.abs(mp - mps[c]) * msk_img
        # axs[1,i].set_title(f'Error {titles[i]} (2X)')
        axs[1,i].imshow(err, cmap='gray', vmin=0, vmax=vmax / 2)
        axs[1,i].axis('off')

        # Show eigenvalues
        # if i == 0:
        #     axs[2,i].set_title('Phantom')
        # else:
        #     axs[2,i].set_title('Eigenvalues')
        axs[2,i].imshow(ev, cmap='gray', vmin=0, vmax=1)
        axs[2,i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
plt.show()