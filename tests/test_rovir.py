import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from einops import rearrange, einsum
from mat73 import loadmat

from mr_recon.utils import gen_grd, np_to_torch
from mr_recon.multi_coil.rovir import apply_rovir
from mr_recon.multi_coil.calib import calc_coil_subspace, synth_cal
from mr_recon.recons import coil_combine
from mr_recon.fourier import fft, ifft

# Params
z_slc = 4
te_slc = 9
torch_dev = torch.device(0)
# torch_dev = torch.device('cpu')

# Load Nan's fully 2D images with coils
fovs = (0.22,)*2
# path = '/local_mount/space/mayday/data/users/Nan/data_GE/Share/20250227_2DEPTI/2D_fullyEPTI.mat'
# data = loadmat(path)
# img = np_to_torch(data['recon_all'][:, :, z_slc, te_slc]).type(torch.complex64).to(torch_dev).moveaxis(-1, 0)
# mps = np_to_torch(data['SEs'][:, :, z_slc]).type(torch.complex64).to(torch_dev).moveaxis(-1, 0)
path = '/local_mount/space/mayday/data/users/abrahamd/coco_fruit/Exam30510/data/'
mps = np_to_torch(np.load(path + 'mps.npy')[..., 0]).type(torch.complex64).to(torch_dev)
img = np_to_torch(np.load(path + 'img.npy')).type(torch.complex64).to(torch_dev)
mps = mps.flip(dims=[1])
img = img.flip(dims=[0])
img /= img.abs().max()
mps /= mps.abs().max()
im_size = img.shape

# Fourier kernel to mps
kern_size = (3, 3)
kerns = gen_grd(kern_size).to(torch_dev).reshape((-1, 2)) * 1.0
rs = gen_grd(im_size).to(torch_dev)
harms = torch.exp(-2j * torch.pi * (rs @ kerns.T)).moveaxis(-1, 0)
mps = (mps[:, None] * harms[None,]).reshape((-1, *im_size))

# Make multi-channel image
coil_imgs = img * mps
ksp_cal = fft(coil_imgs, dim=(-2, -1))
# ksp_cal += 1e-2 * torch.randn_like(ksp_cal)
ksp_cal = synth_cal(ksp_cal, (32, 32))
img_cal = ifft(ksp_cal, dim=(-2, -1), oshape=(mps.shape[0], *im_size))
print(img_cal.abs().max())

# Pick ROI for signal and interference
mask_signal = torch.zeros(im_size, device=torch_dev)
# mask_signal[:, 120:-20] = 1
mask_signal[:-80] = 1
mask_interf = torch.zeros(im_size, device=torch_dev)
mask_interf[-50:] = 1

# Apply ROvir
W, coil_imgs_rovir = apply_rovir(img_cal, mask_signal, mask_interf, 20, 1e-4, coil_imgs)
# W, img_cal_rovir = calc_coil_subspace(img_cal * mask_signal, 1, img_cal)
img_rovir = coil_combine(coil_imgs_rovir)
img = coil_combine(coil_imgs)

# Plot
p = 0.9
plt_tform = lambda x : x * p + (1 - x) * (1 - p)
plt.figure(figsize=(14,7))
plt.subplot(221)
plt.title(f'Image')
plt.imshow(img.abs().cpu(), cmap='gray')
plt.axis('off')
plt.subplot(222)
plt.title(f'Calibration Image')
plt.imshow(coil_combine(img_cal).abs().cpu(), cmap='gray')
plt.axis('off')
plt.subplot(223)
plt.title(f'Rovir Image')
plt.imshow(img_rovir.abs().cpu(), cmap='gray')
plt.axis('off')
plt.subplot(224)
plt.title(f'Rovir Image With Regions')
a = 0.2
img_rovir_rgb = img.abs()[..., None].repeat_interleave(3, dim=-1) / img_rovir.abs().max()
img_rovir_rgb[..., 0] = torch.clamp(img_rovir_rgb[..., 0] * (1 - a) + mask_interf * a, 0, 1)
img_rovir_rgb[..., 1] = torch.clamp(img_rovir_rgb[..., 1] * (1 - a) + mask_signal * a, 0, 1)
plt.imshow(img_rovir_rgb.cpu())
plt.axis('off')
plt.tight_layout()
plt.show()
