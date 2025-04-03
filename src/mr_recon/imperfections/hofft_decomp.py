from mr_recon.algs import eigen_decomp_operator
from mr_recon.linops import type3_nufft
import torch
import numpy as np

from tqdm import tqdm
from typing import Optional
from einops import einsum
from mr_recon.utils import gen_grd, np_to_torch
from mr_recon.dtypes import complex_dtype
from mr_recon.fourier import ifft, fft, sigpy_nufft, triton_nufft

def temporal_eigen(phis: torch.Tensor,
                   alphas: torch.Tensor,):
    """
    Computes just the temporal eigenvectors of the high order phase imperfection.
    
    Args:
    -----
    phis : torch.Tensor
        The spatial phase maps, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    """
    # Consts
    B, *im_size = phis.shape
    B_, *trj_size = alphas.shape
    R = np.prod(im_size)
    T = np.prod(trj_size)
    torch_dev = phis.device
    assert B == B_, "phis and alphas must have same number of bases (B)"
    assert phis.device == alphas.device, "phis and alphas must be on same device"
    
    # Flatten everything
    phis_flt = phis.reshape((B, -1))
    alphas_flt = alphas.reshape((B, -1))
    
    # Center alphas and phis
    phis_mp = (phis_flt.min(dim=1).values + phis_flt.max(dim=1).values)/2
    alphas_mp = (alphas_flt.min(dim=1).values + alphas_flt.max(dim=1).values)/2
    phis_flt -= phis_mp[:, None]
    alphas_flt -= alphas_mp[:, None]
    
    # Rescale phis to be between [-1/2, 1/2]
    scales = phis_flt.abs().max(dim=1).values[:, None] * 2
    phis_flt /= scales
    alphas_flt *= scales
    
    # Determine u-space matrix by extent of alphas
    # breakpoint()
    alphas_range = alphas_flt.abs().max(dim=1).values * 2
    os_factor = 1.0
    umx_size = tuple((os_factor * alphas_range).ceil().int().tolist())
    ku_grd = gen_grd(umx_size, umx_size).to(torch_dev) # *umx_size, B
    ku_grd = ku_grd.reshape((-1, B))
    U = np.prod(umx_size)
    print(f'U Matrix Size = {umx_size}, {U} total points')
    print(f'Size of LLH is {8 * (U ** 2) / 2 ** 30:.3f} GB')
    
    # Build LLH matrix, very slow for now
    LLH = torch.zeros((U, U), device=torch_dev, dtype=complex_dtype)
    bs = 2 ** 12
    for r1 in tqdm(range(0, R, bs), 'Building LLH'):
        r2 = min(r1 + bs, R)
        u_vec = phis_flt[:, r1:r2] # B R
        phz = ku_grd @ u_vec # U R
        phz = phz.reshape((*umx_size, r2-r1))
        delta = ifft(torch.exp(-2j * torch.pi * phz),
                     dim=tuple(range(B)),
                     norm='backward') # *umx_size R
        delta = delta.reshape((U, r2-r1)) # U R
        LLH += delta @ delta.H

    # Build linop, and verify
    nufft = sigpy_nufft(umx_size)
    def AAH(h):
        # h is ... T
        Fh = nufft.adjoint(h[None,], alphas_flt.T[None,])[0] # ... *umx_size
        Fh = Fh.reshape((*Fh.shape[:-B], -1)) # ... U
        LLHFh = einsum(LLH, Fh, 'Uo Ui, ... Ui -> ... Uo') # ... U
        LLHFh = LLHFh.reshape((*LLHFh.shape[:-1], *umx_size)) # ... *umx_size
        FLLHFh = nufft(LLHFh[None,], alphas_flt.T[None,])[0] # ... T
        return FLLHFh
    def AAH_gt(h):
        # h is ... T
        phz = alphas_flt.T @ phis_flt # T R
        a = torch.exp(-2j * np.pi * phz) # T R
        aah = a @ a.H # T T
        return einsum(aah, h, 'To Ti, ... Ti -> ... To') # ... T

    # Eigen decomposition
    x0 = torch.randn(T, device=torch_dev, dtype=complex_dtype) # Initial guess
    vecs, vals = eigen_decomp_operator(AAH, x0, num_eigen=2, num_iter=100, lobpcg=True)
    
    return vecs.reshape((-1, *trj_size))

def temporal_psf_eigen(phis: torch.Tensor,
                       alphas: torch.Tensor,):
    """
    Computes the temporal eigenvectors of the high order phase imperfection
    using a PSF approach.
    
    Args:
    -----
    phis : torch.Tensor
        The spatial phase maps, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    """
    # Consts
    B, *im_size = phis.shape
    B_, *trj_size = alphas.shape
    R = np.prod(im_size)
    T = np.prod(trj_size)
    torch_dev = phis.device
    assert B == B_, "phis and alphas must have same number of bases (B)"
    assert phis.device == alphas.device, "phis and alphas must be on same device"
    
    # Flatten everything
    phis_flt = phis.reshape((B, R))
    alphas_flt = alphas.reshape((B, T))
    
    # Manually
    inp = torch.randn(phis_flt.shape[1], dtype=torch.complex64, device=torch_dev)
    inp_t = torch.randn(alphas_flt.shape[1], dtype=torch.complex64, device=torch_dev)
    
    # Test linop
    def A_gt(x):
        phz = alphas_flt.T @ phis_flt # T R
        a = torch.exp(-2j * np.pi * phz) # T R
        return a @ x
    def AH_gt(y):
        phz = alphas_flt.T @ phis_flt # T R
        a = torch.exp(-2j * np.pi * phz) # T R
        return a.H @ y
    
    t3n = type3_nufft(phis_flt, alphas_flt, oversamp=2.0, width=4, use_toep=True)
    
    eigen_decomp_operator(lambda x : t3n.normal(x), torch.randn_like(inp), num_eigen=20, num_iter=100)
    quit()
    
    out = t3n.forward(inp[None,])[0]
    out_gt = A_gt(inp)
    # out = t3n.adjoint(inp_t[None,])[0]
    # out_gt = AH_gt(inp_t)
    scale = (out.conj() * out_gt).sum() / (out.conj() * out).sum()
    out *= scale
    
    import matplotlib.pyplot as plt
    
    plt.plot(out.real.cpu())
    # plt.figure()
    plt.plot(out_gt.real.cpu(), ls='--')
    plt.show()
    quit()
    
    
