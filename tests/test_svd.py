import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.algs import svd_power_method_tall, svd_operator

# Gen data
device = torch.device(4)
A = torch.randn((100, 512), device=device, dtype=torch.complex64)
rank = torch.linalg.matrix_rank(A)

# Run svd with torch
u_torch, s_torch, vh_torch = torch.linalg.svd(A, full_matrices=False)

# Try ours
AHA = A.H @ A
A_op = lambda x : A @ x
AHA_op = lambda x : AHA @ x
inp_example = torch.randn((A.shape[1],), device=device, dtype=A.dtype)
u, s, vh = svd_operator(A=A_op,
                        AHA=AHA_op,
                        inp_example=inp_example,
                        rank=rank,
                        # tol=1e-2,
                        num_iter=15)
# u, s, vh = svd_power_method_tall(A=A_op,
#                                  AHA=AHA_op,
#                                  inp_dims=inp_example.shape,
#                                  inp_dtype=inp_example.dtype,
#                                  rank=rank,
#                                  device=device)

# Show results
A_est_torch = (s_torch * u_torch) @ vh_torch
A_est = (s * u) @ vh
print(f'Error Torch = {torch.linalg.norm(A_est_torch - A).cpu()}')
print(f'Error Ours  = {torch.linalg.norm(A_est - A).cpu()}')