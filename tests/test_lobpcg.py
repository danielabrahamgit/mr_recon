import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from einops import einsum
from mr_recon.algs import eigen_decomp_operator


n = 100
k = 100
A = torch.randn((n, n), dtype=torch.complex64)
A = A.H @ A # 

X = torch.randn((n, k), dtype=A.dtype)


A_op = lambda x: A @ x
evecs, evals = eigen_decomp_operator(A_op, X[:, 0], num_eigen=k, num_iter=10, lobpcg=True)
evecs = evecs.T
evals = evals.real
inds = torch.argsort(evals, descending=True)
evals = evals[inds]
evecs = evecs[:, inds]
eest = ((evecs * evals) @ evecs.H)

# vals, vecs = torch.linalg.eigh(A)
vecs, vals = eigen_decomp_operator(A_op, X[:, 0], num_eigen=k, num_iter=100, lobpcg=False)
vecs = vecs.T
vals = vals.real
inds = torch.argsort(vals, descending=True)
vals = vals[inds]
vecs = vecs[:, inds]
est = ((vecs * vals) @ vecs.H)


print((A - eest).norm())
print((A - est).norm())

# plt.figure()
# plt.imshow(A.real)

# plt.figure()
# plt.imshow(eest.real)

# plt.figure()
# plt.imshow(est.real)


# plt.show()