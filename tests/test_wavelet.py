import torch

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_recon.prox import L1Wav_cpu, L1Wav
from tqdm import tqdm

# torch_dev = torch.device(6)
torch_dev = torch.device('cpu')
im_size = (256,)*2
# im_size = (768, 768, 112)
img = shepp_logan(torch_dev).img(im_size)

lamda = 0.1
rnd_shift = 0
rnd_phase = False
W = L1Wav(im_size, lamda=lamda, rnd_shift=rnd_shift, rnd_phase=rnd_phase)
Wcpu = L1Wav_cpu(im_size, lamda=lamda, rnd_shift=rnd_shift, rnd_phase=rnd_phase)
N = 1
for _ in tqdm(range(N), desc='GPU Prox'):
    img_gpu = W(img)
for _ in tqdm(range(N), desc='CPU Prox'):
    img_cpu = Wcpu(img)
    
if len(im_size) == 2:
    slc = slice(None)
else:
    slc = (slice(None), slice(None), im_size[2] // 2)

plt.figure(figsize=(14, 7))
plt.subplot(131)
plt.imshow(img[slc].abs().rot90().cpu(), cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(132)
plt.imshow(img_cpu[slc].abs().rot90().cpu(), cmap='gray')
plt.title('CPU Prox')
plt.axis('off')
plt.subplot(133)
plt.imshow(img_gpu[slc].abs().rot90().cpu(), cmap='gray')
plt.title('GPU Prox')
plt.axis('off')
plt.tight_layout()
plt.show()