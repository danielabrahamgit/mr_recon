import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.algs import svd_power_method_tall

# Gen data
device = torch.device(4)
A = torch.randn((10, 12), device=device)
rank = torch.linalg.matrix_rank(A)

# Run svd with torch
u_torch, s_torch, vh_torch = torch.linalg.svd(A, full_matrices=False)

# Try ours
AHA = A.H @ A
A_op = lambda x : A @ x
AHA_op = lambda x : AHA @ x
u, s, vh = svd_power_method_tall(A=A_op,
                                 AHA=AHA_op,
                                 inp_dims=A.shape[-1:],
                                 rank=rank,
                                 device=device)

# Show results
A_est_torch = (s_torch * u_torch) @ vh_torch
A_est = (s * u) @ vh
print(f'Error Torch = {torch.linalg.norm(A_est_torch - A).cpu()}')
print(f'Error Ours  = {torch.linalg.norm(A_est - A).cpu()}')
plt.figure(figsize=(14,7))
plt.subplot(131)
plt.imshow(A.cpu())
plt.title('A')
plt.axis('off')
plt.subplot(132)
plt.imshow(A_est_torch.cpu())
plt.title('A Est Torch')
plt.axis('off')
plt.subplot(133)
plt.imshow(A_est.cpu())
plt.title('A Est Ours')
plt.axis('off')
plt.tight_layout()
plt.show()