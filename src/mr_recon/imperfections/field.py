import torch
import numpy as np

from typing import Optional
from einops import rearrange, einsum
from tqdm import tqdm
from mr_recon.dtypes import real_dtype, complex_dtype
from mr_recon.utils import quantize_data
from mr_recon.linops import type3_nufft, type3_nufft_naive
from mr_recon.algs import svd_operator

def coco_bases(x, y, z):
    assert x.shape == y.shape
    assert z.shape == x.shape
    tup = (None,) + (slice(None),) * x.ndim
    x = x[tup]
    y = y[tup]
    z = z[tup]
    return torch.cat([
        z * z,
        x * x + y * y,
        x * z,
        y * z
    ], dim=0)

def sph_bases(x, y, z):
    assert x.shape == y.shape
    assert z.shape == x.shape
    tup = (None,) + (slice(None),) * x.ndim
    x = x[tup]
    y = y[tup]
    z = z[tup]
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    x3 = x ** 3
    y3 = y ** 3
    z3 = z ** 3
    return torch.cat([
        torch.ones_like(x),
        x,
        y,
        z,
        x * y,
        z * y,
        3 * z2 - (x2 + y2 + z2),
        x * z,
        x2 - y2,
        3 * y * x2 - y3, 
        x * y * z,
        (5 * z2 - (x2 + y2 + z2)) * y,
        5 * z3 - 3 * z * (x2 + y2 + z2),
        (5 * z2 - (x2 + y2 + z2)) * x,
        z * x2 - z * y2,
        x3 - 3 * x * y2
    ], dim=0)

def b0_to_phis_alphas(b0_map: torch.Tensor,
                      trj_size: tuple,
                      ro_dim: int,
                      dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert B0 field map to phi and alpha notation

    Args:
    -----
    b0_map : torch.Tensor
        B0 field map in Hz with shape (*im_size)
    trj_size : tuple
        Size of the trajectory, arbitrary dimensions
    ro_dim : int
        Readout dimension, but be in [0, len(trj_size))
    dt : float
        Time step in seconds

    Returns:
    --------
    phis : torch.Tensor
        Phase basis in radians with shape (1, *im_size),
        normalized to range [-1/2, 1/2]
    alphas : torch.Tensor
        Phase coefficients with shape (1, *trj_size)
    """
    
    # Normalize b0_map to range [-1/2, 1/2]
    scale = 2 * b0_map.abs().max()
    # scale = 1.0
    phis = b0_map / scale
    
    # Make alphas
    ts = torch.arange(trj_size[ro_dim], device=b0_map.device, dtype=real_dtype) * dt * scale
    tup = (slice(None),) + (None,) * (len(trj_size) - 1)
    alphas = ts[tup].moveaxis(0, ro_dim)
    
    # Return
    return phis[None,], alphas[None,]
    
def coco_to_phis_alphas(trj: torch.Tensor,
                        spatial_crds: torch.Tensor,
                        field_strength: float,
                        ro_dim: int,
                        dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert trajectory to concomitant fields phi and alpha notation

    Args:
    -----
    trj : torch.Tensor
        K-space trajectory with shape (*trj_size, 3) in units of 1/meters
    spatial_crds : torch.Tensor
        Spatial coordinates with shape (*im_size, 3) in units of meters
    field_strength : float
        Field strength in units of Teslas
    ro_dim : int
        Readout dimension, but be in [0, len(trj_size))
    dt : float
        Sampling time along readout in units of seconds

    Returns:
    --------
    phis : torch.Tensor
        Phase basis in radians with shape (4, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (4, *trj_size)
    """
    # Consts
    trj = trj.swapaxes(0, ro_dim)
    d = trj.shape[-1]
    trj_size = trj.shape[:-1]
    gamma_bar = 42.5774e6 # Hz / T
    assert d == 3
    assert d == spatial_crds.shape[-1]

    # Get gradient from trj
    trj = trj.type(real_dtype)
    g = torch.diff(trj, dim=0) / (dt * gamma_bar)
    g = torch.cat((g, g[-1:]), dim=0)
    
    # Build phis and alphas
    alphas = torch.zeros((4, *trj_size), dtype=trj.dtype, device=trj.device)
    X, Y, Z = spatial_crds[..., 0], spatial_crds[..., 1], spatial_crds[..., 2]
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    phis = coco_bases(X, Y, Z)
    alphas[0] = gx ** 2 + gy ** 2
    alphas[1] = (gz ** 2) / 4
    alphas[2] = -gx * gz 
    alphas[3] = -gy * gz
    alphas /= 2 * field_strength

    # Integral on alphas, gamma_bar to map T to phase
    alphas = torch.cumulative_trapezoid(alphas, dx=dt, dim=1) * gamma_bar
    alphas = torch.cat([alphas[:, :1] * 0, alphas], dim=1)
    
    return phis, alphas.swapaxes(1, ro_dim+1)

def rescale_phis_alphas(phis: torch.Tensor,
                        alphas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase model is 
    phase(r, t) = 2pi sum_k phi_k(r) * alpha_k(t)
    
    We will use the following rescaling:
    phase(r,t) = 2pi sum_k (phi_nrm_k(r) + phi_mp_k) * (alpha_nrm(t) + alpha_mp_k)
    
    where phi_nrm are normalized to be between [-1/2, 1/2], meaning that alpha_nrm 
    tel you how many phase wraps accumulate
    
    Args:
    -----
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
        
    Returns:
    --------
    phis_nrm : torch.Tensor
        Normalized phase basis with shape (B, *im_size)
    phis_mp : torch.Tensor
        Phase basis midpoints with shape (B,)
    alphas_nrm : torch.Tensor
        Normalized phase coefficients with shape (B, *trj_size)
    alphas_mp : torch.Tensor
        Phase coefficients midpoints with shape (B,)
    """
    # Consts
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    B = phis.shape[0]
    R = np.prod(im_size)
    T = np.prod(trj_size)
    assert B == alphas.shape[0]
    
    # Flatten everything
    phis_flt = phis.reshape((B, R))
    alphas_flt = alphas.reshape((B, T))
    
    # Center alphas and phis
    phis_mp = (phis_flt.min(dim=1).values + phis_flt.max(dim=1).values)/2
    alphas_mp = (alphas_flt.min(dim=1).values + alphas_flt.max(dim=1).values)/2
    phis_flt_cent = phis_flt - phis_mp[:, None]
    alphas_flt_cent = alphas_flt - alphas_mp[:, None]
    
    # Rescale phis to be between [-1/2, 1/2]
    scales = phis_flt_cent.abs().max(dim=1).values * 2
    phis_flt_cent /= scales[:, None]
    phis_mp /= scales
    alphas_flt_cent *= scales[:, None]
    alphas_mp *= scales
    
    # Reshape and return
    phis_nrm = phis_flt_cent.reshape((B, *im_size))
    alphas_nrm = alphas_flt_cent.reshape((B, *trj_size))
    return phis_nrm, phis_mp, alphas_nrm, alphas_mp
    
def isotropic_cluster_alphas(alphas: torch.Tensor,
                             phis: torch.Tensor,
                             L: int) -> torch.Tensor:
    """
    Clusters alphas in an isotropic way, removing midpoints first
    
    Args:
    -----
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    L : int
        Number of clusters
        
    Returns:
    --------
    alpha_cents : torch.Tensor
        Cluster centers with shape (L, B)
    inds : torch.Tensor
        Cluster indices with shape (*trj_size) in [0, L)
    """
    # Consts
    B = alphas.shape[0]
    
    # Flatten everything
    trj_size = alphas.shape[1:]
    phis_flt = phis.reshape((B, -1))
    alphas_flt = alphas.reshape((B, -1))
    
    # Center alphas and phis
    phis_mp = (phis_flt.min(dim=1).values + phis_flt.max(dim=1).values)/2
    alphas_mp = (alphas_flt.min(dim=1).values + alphas_flt.max(dim=1).values)/2
    phis_flt_cent = phis_flt - phis_mp[:, None]
    alphas_flt_cent = alphas_flt - alphas_mp[:, None]
    
    # Rescale phis to be between [-1/2, 1/2]
    scales = phis_flt_cent.abs().max(dim=1).values * 2
    phis_flt_cent /= scales[:, None]
    phis_mp /= scales
    alphas_flt_cent *= scales[:, None]
    alphas_mp *= scales
    
    # Cluster centered alphas
    alpha_cents, inds = quantize_data(alphas_flt_cent.T, L, method='cluster')
    
    # Rescale back
    alpha_cents /= scales
    alpha_cents += alphas_mp / scales
    
    # Reshape
    alpha_cents = alpha_cents.reshape((L, B))
    inds = inds.reshape(*trj_size)
    
    return alpha_cents, inds

def alpha_segementation(phis: torch.Tensor,
                        alphas: torch.Tensor,
                        L: int,
                        L_batch_size: Optional[int] = 1,
                        interp_type: Optional[str] = 'zero',
                        use_type3: Optional[bool] = True,
                        manual_spatial_funcs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs the following decomposition:
    
    e^{-j 2pi phis(r) * alphas(t)} = sum_{l=0}^{L-1} b_l(r) * h_l(t)
    
    where
    b_l(r) = e^{-j 2pi phis(r) * alphas_l}
    h_l(t) are the interpolation funcitons
    
    alphas_l are chosen according to k-means clustering 

    Parameters:
    -----------
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
    L : int
        Number of decomposition components
    interp_type : str
        'zero' - zero order interpolator
        'linear' - linear interpolator 
        'lstsq' - least squares interpolator
    L_batch_size : int
        Batch size for applying high order transform
    use_type3 : bool
        If True, use type3 nufft for forward pass

    Returns:
    --------
    spatial_funcs : torch.Tensor
        Spatial basis functions with shape (L, *im_size)
    temporal_funcs : torch.Tensor
        Temporal basis functions with shape (L, *trj_size)
    """
    # Consts
    torch_dev = phis.device
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    B = phis.shape[0]
    assert B == alphas.shape[0]
    
    # Get alpha clusters
    if B == 1:
        alpha_cents, inds = quantize_data(alphas.moveaxis(0, -1), L, method='uniform')
    else:
        alpha_cents, inds = isotropic_cluster_alphas(alphas, phis, L)
        # alpha_cents, inds = quantize_data(alphas.moveaxis(0, -1), L, method='cluster')

    # Compute spatial basis functions
    spatial_funcs = torch.zeros(L, *im_size, device=torch_dev, dtype=complex_dtype)
    for i in range(L):
        spatial_funcs[i, ...] = torch.exp(-2j * torch.pi * (phis.moveaxis(0, -1) @ alpha_cents[i]))
    
    # TODO Debug
    if manual_spatial_funcs is not None:
        spatial_funcs = manual_spatial_funcs

    # Compute temporal basis functions
    temporal_funcs = torch.zeros(L, *trj_size, device=torch_dev, dtype=complex_dtype)
    if 'zero' in interp_type:
        for i in range(L):
            temporal_funcs[i, ...] = ((inds == i) * 1.0)
    elif 'linear' in interp_type:
        raise NotImplementedError
    elif 'lstsq' in interp_type:
        # Create type3 nufft for forward pass
        if use_type3:
            t3n = type3_nufft(phis, alphas)
        else:
            t3n = type3_nufft_naive(phis, alphas)
        
        # First compute AHA, which is all the pairwise 
        # dot products of the spatial features
        AHA = torch.zeros((L, L), device=torch_dev, dtype=complex_dtype)
        for i in range(L):
            for j in range(L):
                AHA[i, j] = (spatial_funcs[i].conj() * spatial_funcs[j]).mean()
                
        # Next compute AHy, which is essentially int_r W(r, t) * b_l(r).conj() dr
        AHy = torch.zeros((L, *trj_size), device=torch_dev, dtype=complex_dtype)
        for l1 in tqdm(range(0, L, L_batch_size), 'Least Squares Forward Pass'):
            l2 = min(l1 + L_batch_size, L)
            AHy[l1:l2, ...] = t3n.forward(spatial_funcs[l1:l2].conj()) / np.prod(im_size)

        # Solve for h_l(t) using least squares
        AHA_inv = torch.linalg.pinv(AHA)
        temporal_funcs = einsum(AHA_inv, AHy, 'Lo Li, Li ... -> Lo ...')

    return spatial_funcs, temporal_funcs

def alpha_phi_svd(phis: torch.Tensor,
                  alphas: torch.Tensor,
                  L: int,
                  mask: Optional[torch.Tensor] = None,
                  L_batch_size: Optional[int] = 1,
                  num_iter: Optional[int] = 15,
                  use_type3: Optional[bool] = True,
                  verbose: Optional[bool] = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs the following decomposition:
    
    e^{-j 2pi phis(r) * alphas(t)} = sum_{l=0}^{L-1} b_l(r) * h_l(t)
    
    where
    b_l(r) are SVD computed spatial basis functions
    h_l(t) are SVD computed temporal basis functions

    Parameters:
    -----------
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
    L : int
        Number of decomposition components
    mask : torch.Tensor
        Image mask with shape im_size
    L_batch_size : int
        Batch size for applying high order transform
    num_iter : int
        Number of iterations for LOBPCG part of SVD computation
    use_type3 : bool
        If True, use type3 nufft for forward pass
    verbose : bool
        If True, print progress

    Returns:
    --------
    spatial_funcs : torch.Tensor
        Spatial basis functions with shape (L, *im_size)
    temporal_funcs : torch.Tensor
        Temporal basis functions with shape (L, *trj_size)
    """    
    # Consts
    torch_dev = phis.device
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    B = phis.shape[0]
    assert B == alphas.shape[0]
    
    # Make type3 nufft with toeplitz
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)
        
    # Default mask
    if mask is None:
        mask = torch.ones(im_size, device=torch_dev, dtype=complex_dtype)
    
    # Batched implimentations of forward and normal operators
    def forward(x):
        # x has shape (L, *im_size)
        ys = []
        for l1 in range(0, x.shape[0], L_batch_size):
            l2 = min(l1 + L_batch_size, x.shape[0])
            ys.append(t3n.forward(x[l1:l2] * mask))
        return torch.cat(ys, dim=0)
    def normal(x):
        # x has shape (L, *trj_size)
        ys = []
        for l1 in range(0, x.shape[0], L_batch_size):
            l2 = min(l1 + L_batch_size, x.shape[0])
            ys.append(t3n.normal(x[l1:l2] * mask) * mask)
        return torch.cat(ys, dim=0)
    
    # SVD 
    inp_example = torch.randn(im_size, device=torch_dev, dtype=complex_dtype)
    U, S, Vh = svd_operator(forward, normal, inp_example, rank=L, num_iter=num_iter, lobpcg=True, verbose=verbose)
    
    # Reshape U and Vh to be spatial and temporal basis functions
    temporal_funcs = (U * (S ** 0.5)).moveaxis(-1, 0)
    spatial_funcs = (Vh.moveaxis(0, -1) * (S ** 0.5)).moveaxis(-1, 0)
    
    return spatial_funcs, temporal_funcs
 
def alpha_phi_svd_with_grappa(phis: torch.Tensor,
                              alphas: torch.Tensor,
                              kerns: torch.Tensor,
                              mps: torch.Tensor,
                              L: int,
                              L_batch_size: Optional[int] = 1,
                              use_type3: Optional[bool] = True,
                              num_iter: Optional[int] = 15,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs the following decomposition:
    
    e^{-j 2pi phis(r) * alphas(t)} = sum_{l=0}^{L-1} b_l(r) * h_l(t)
    
    where
    b_l(r) are SVD computed spatial basis functions
    h_l(t) are SVD computed temporal basis functions

    Parameters:
    -----------
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
    kerns : torch.Tensor
        GRAPPA kernels with shape (C, C, *trj_size)
    mps : torch.Tensor
        Coil sensivity maps (or coil wieghted calib images) with shape (C, *im_size)
    L : int
        Number of decomposition components
    L_batch_size : int
        Batch size for applying high order transform
    use_type3 : bool
        If True, use type3 nufft for forward pass
    num_iter : int
        Number of iterations for LOBPCG part of SVD computation

    Returns:
    --------
    spatial_funcs : torch.Tensor
        Spatial basis functions with shape (L, *im_size)
    temporal_funcs : torch.Tensor
        Temporal basis functions with shape (L, C, *trj_size)
    """    
    # Consts
    torch_dev = phis.device
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    B = phis.shape[0]
    C = mps.shape[0]
    assert B == alphas.shape[0]
    assert C == kerns.shape[0]
    assert C == kerns.shape[1]
    
    # Make type3 nufft with no toeplitz
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=False)
    else:
        t3n = type3_nufft_naive(phis, alphas)
    
    # Batched implimentations of forward and normal operators
    def forward(x):
        # x has shape (L, *im_size)
        ys = []
        for l1 in range(0, x.shape[0], L_batch_size):
            l2 = min(l1 + L_batch_size, x.shape[0])
            x_mps = (x[l1:l2, None] * mps[None, :]).reshape((-1, *im_size))
            y = t3n.forward(x_mps) # ((L C), *trj_size)
            y = y.reshape((l2-l1, C, *trj_size))
            y_kern = einsum(kerns, y, 'Co Ci ..., L Ci ... -> L Co ...')
            ys.append(y_kern)
        return torch.cat(ys, dim=0)
    def adjoint(y):
        # y has shape (L, C, *trj_size)
        xs = []
        for l1 in range(0, y.shape[0], L_batch_size):
            l2 = min(l1 + L_batch_size, y.shape[0])
            y_kern = einsum(kerns.conj(), y[l1:l2], 'Co Ci ..., L Co ... -> L Ci ...')
            y_kern = y_kern.reshape((-1, *trj_size))
            x_mps = t3n.adjoint(y_kern) # ((L C), *im_size)
            x_mps = x_mps.reshape((l2-l1, C, *im_size))
            x = einsum(x_mps, mps.conj(), 'L C ..., C ... -> L ...')
            xs.append(x)
        return torch.cat(xs, dim=0)
    def normal(x):
        return adjoint(forward(x))
    
    # SVD 
    inp_example = torch.randn(im_size, device=torch_dev, dtype=complex_dtype)
    U, S, Vh = svd_operator(forward, normal, inp_example, rank=L, num_iter=num_iter, lobpcg=True)
    
    # Reshape U and Vh to be spatial and temporal basis functions
    temporal_funcs = (U * (S ** 0.5)).moveaxis(-1, 0)
    spatial_funcs = (Vh.moveaxis(0, -1) * (S ** 0.5)).moveaxis(-1, 0)
    
    return spatial_funcs, temporal_funcs
    