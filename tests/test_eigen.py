import torch
import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt
from mr_recon.algs import eigen_decomp_operator

n = 20
torch_dev = torch.device(5)

# Build simple matrix
A = torch.randn(n, n, device=torch_dev)
U, s, Vh = torch.linalg.svd(A)
s = 1 / torch.arange(1, n+1, device=torch_dev).float() ** 2
A = U @ (s[None, :] * U.H)

# Compute eigenvecs/vals
x0 = torch.randn(n, device=torch_dev)
vecs, vals = eigen_decomp_operator(A=lambda x : A @ x, x0=x0, num_eigen=10, num_power_iter=150, reverse_order=True)

plt.plot(1 / s.cpu().flip(0))
plt.plot(vals.cpu())
plt.show()