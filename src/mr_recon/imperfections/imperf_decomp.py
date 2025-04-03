import torch

from typing import Optional
from mr_recon.imperfections.spatio_temporal_imperf import spatio_temporal
from mr_recon.algs import svd_power_method_tall, svd_matrix_method_tall, lin_solve
from mr_recon.dtypes import complex_dtype
from mr_recon.utils import gen_grd
from einops import rearrange, einsum
from tqdm import tqdm

"""
All of these techniques are used to decompose a spatio-temporal imperfection:
    
    W(r, p(t)) = sum_{l=1}^{L} h_l(t) b_l(r)
"""

def decomp_with_grappa(st_imperf: spatio_temporal,
                       kerns: torch.Tensor,
                       mps: torch.Tensor,
                       L: int,
                       niter: Optional[int] = 50,
                       method: Optional[str] = 'svd',):
    """
    Decomposes residual imperfection, where some of the imperfection was removed by GRAPPA kernels.
    
    Parameters:
    -----------
    st_imperf : spatio_temporal
        spatio temporal imperfection
    kerns : torch.Tensor
        GRAPPA kernels with shape (C, C, *trj_size)
    mps : torch.Tensor
        Multi-coil sensitivity maps with shape (C, *im_size)
    L : int
        Number of decomposition components
    niter : int
        Number of iterations for power method
    method : str
        'svd' - SVD formulation
        'ts' - time segementation formulation
    """
    def forward(x):
        Ax = st_imperf.forward_matrix_prod(x * mps)
        y = einsum(kerns, Ax, 'Co Ci ..., Ci ... -> Co ...')
        return y
    def adjoint(y):
        Ghy = einsum(kerns.conj(), y, 'Co Ci ..., Co ... -> Ci ...')
        AhGhy = st_imperf.adjoint_matrix_prod(Ghy)
        x = einsum(mps.conj(), AhGhy, 'C ..., C ... -> ...')
        return x
    def normal_spatial(x):
        return adjoint(forward(x))
    def normal_temporal(y):
        return forward(adjoint(y))
    
    # Consts
    im_size = st_imperf.im_size
    trj_size = st_imperf.trj_size
    C = mps.shape[0]
    assert kerns.shape[0] == kerns.shape[1], "Kernels must be square"
    assert list(kerns.shape[2:]) == list(trj_size), "Kernels must match trajectory size"
    assert mps.shape[0] == kerns.shape[0], "MPS and kernels must have same coils"
    assert list(mps.shape[1:]) == list(im_size), "MPS must match image size"
    
    if method == 'ts':
        return None
    else:
        # SVD formulation
        U, S, Vh = svd_power_method_tall(A=forward,
                                         AHA=normal_spatial,
                                         inp_dims=im_size,
                                         rank=L,
                                         niter=niter,
                                         device=st_imperf.torch_dev)
        p = 0.5
        temporal_funcs = rearrange(U * (S ** p), '... L -> L ...')
        spatial_funcs = einsum(Vh, (S ** (1 - p)), 'L ..., L -> L ...')
        
    return spatial_funcs, temporal_funcs
        
def svd_decomp_matrix(st_imperf: spatio_temporal,
                      L: int) -> torch.Tensor:
    """
    Perform SVD decomposition of the spatio temporal imperfection matrix.

    Picks h_l(t) and b_l(r) using SVD.

    Parameters:
    -----------
    st_imperf : spatio_temporal
        spatio temporal imperfection
    L : int
        Number of decomposition components

    Returns:
    --------
    spatial_funcs : torch.Tensor
        Spatial basis functions with shape (L, *im_size)
    temporal_funcs : torch.Tensor
        Temporal basis functions with shape (L, *trj_size)
    """
    # Gen all spatial inds
    im_size = st_imperf.im_size
    spatial_inds = gen_grd(im_size, im_size) + torch.tensor(im_size) // 2
    spatial_inds = spatial_inds.type(torch.int).reshape((-1, len(im_size))).T
    spatial_inds = spatial_inds.to(st_imperf.torch_dev)

    # Gen all temporal inds
    trj_size = st_imperf.trj_size
    trj_inds = gen_grd(trj_size, trj_size) + torch.tensor(trj_size) // 2
    trj_inds = trj_inds.type(torch.int).reshape((-1, len(trj_size))).T
    trj_inds = trj_inds.to(st_imperf.torch_dev)
    temporal_features = st_imperf.temporal_features(trj_inds)

    # Access matrix
    spatial_inds = spatial_inds[:, None, :] # d 1 nvox
    temporal_features = temporal_features[:, :, None] # f ntrj 1
    mat = st_imperf.matrix_access(spatial_inds, temporal_features)
    
    # SVD
    U, S, Vh = svd_matrix_method_tall(mat, rank=L)
    temporal_funcs = (U[:, :L] * (S[None, :L] ** 0.5)).T.reshape((L, *trj_size))
    spatial_funcs = (Vh[:L, :] * (S[:L, None] ** 0.5)).reshape((L, *im_size))

    return spatial_funcs, temporal_funcs

def svd_decomp_operator(st_imperf: spatio_temporal,
                        L: int,
                        niter: int = 50,
                        fast_axis: Optional[str] = None) -> torch.Tensor:
    """
    Perform SVD decomposition of the spatio temporal imperfection,
    specifically with an operator formulation.

    Picks h_l(t) and b_l(r) using SVD.

    Parameters:
    -----------
    st_imperf : spatio_temporal
        spatio temporal imperfection
    L : int
        Number of decomposition components
    niter : int
        Number of iterations for power method
    fast_axis : str
        'temporal' - temporal axis is the fast axis
        'spatial' - spatial axis is the fast axis

    Returns:
    --------
    spatial_funcs : torch.Tensor
        Spatial basis functions with shape (L, *im_size)
    temporal_funcs : torch.Tensor
        Temporal basis functions with shape (L, *trj_size)
    """

    nvox = torch.prod(torch.tensor(st_imperf.im_size)).item()
    ntrj = torch.prod(torch.tensor(st_imperf.trj_size)).item()
    if fast_axis is None:
        if nvox < ntrj:
            fast_axis = 'spatial'
        else:
            fast_axis = 'temporal'

    if 'temporal' in fast_axis:
        A = lambda x : st_imperf.adjoint_matrix_prod(x[None,])[0]
        AHA = lambda x : st_imperf.gram_temporal_prod(x[None,])[0]
        U, S, Vh = svd_power_method_tall(A=A,
                                         AHA=AHA,
                                         inp_dims=st_imperf.trj_size,
                                         rank=L,
                                         niter=niter,
                                         device=st_imperf.torch_dev)
        spatial_funcs = rearrange(U * (S ** 0.5), '... L -> L ...').conj()
        temporal_funcs = einsum(Vh, (S ** 0.5), 'L ..., L -> L ...').conj()
    elif 'spatial' in fast_axis:
        A = lambda x : st_imperf.forward_matrix_prod(x[None,])[0]
        AHA = lambda x : st_imperf.gram_spatial_prod(x[None,])[0]
        U, S, Vh = svd_power_method_tall(A=A,
                                         AHA=AHA,
                                         inp_dims=st_imperf.im_size,
                                         rank=L,
                                         niter=niter,
                                         device=st_imperf.torch_dev)
        temporal_funcs = rearrange(U * (S ** 0.5), '... L -> L ...')
        spatial_funcs = einsum(Vh, (S ** 0.5), 'L ..., L -> L ...')


    return spatial_funcs, temporal_funcs

def temporal_segmentation(st_imperf: spatio_temporal,
                          L: int,
                          interp_type: Optional[str] = 'zero',
                          L_batch_size: Optional[int] = None) -> torch.Tensor:
    """
    Performs temporal segmentation of the spatio temporal imperfection.

    Picks b_l(r) = W(r, p[l]) and h_l(t) is a temporal interplator

    Parameters:
    -----------
    st_imperf : spatio_temporal
        spatio temporal imperfection
    L : int
        Number of decomposition components
    interp_type : str
        'zero' - zero order interpolator
        'linear' - linear interpolator 
        'lstsq' - least squares interpolator
    L_batch_size : int
        Batch size for least squares interpolation, defaults to L

    Returns:
    --------
    spatial_funcs : torch.Tensor
        Spatial basis functions with shape (L, *im_size)
    temporal_funcs : torch.Tensor
        Temporal basis functions with shape (L, *trj_size)
    """
    
    # Request clusters of p(t)
    clusters, inds = st_imperf.get_temporal_clusters(L)

    # Compute spatial basis functions
    im_size = st_imperf.im_size
    spatial_funcs = torch.zeros(L, *im_size, device=st_imperf.torch_dev, dtype=complex_dtype)
    grd = gen_grd(im_size, im_size).reshape((-1, len(im_size)))
    grd = (grd + torch.tensor(im_size)//2).type(torch.int).T
    grd = grd.to(st_imperf.torch_dev)
    for i in range(L):
        temp_features = torch.tile(clusters[:, None, i], (1, grd.shape[1]))
        spatial_funcs[i, ...] = st_imperf.matrix_access(grd, temp_features).reshape(im_size)

    # Compute temporal basis functions
    temporal_funcs = torch.zeros(L, *st_imperf.trj_size, device=st_imperf.torch_dev, dtype=complex_dtype)
    if 'zero' in interp_type:
        for i in range(L):
            temporal_funcs[i, ...] = ((inds == i) * 1.0)
    elif 'linear' in interp_type:
        raise NotImplementedError
    elif 'lstsq' in interp_type:
        # First compute AHA, which is all the pairwise 
        # dot products of the spatial features
        AHA = torch.zeros((L, L), device=st_imperf.torch_dev, dtype=complex_dtype)
        for i in range(L):
            for j in range(L):
                AHA[i, j] = (spatial_funcs[i].conj() * spatial_funcs[j]).mean()
                
        # Next compute AHy, which is essentially int_r W(r, t) * b_l(r).conj() dr
        AHy = torch.zeros((L, *st_imperf.trj_size), device=st_imperf.torch_dev, dtype=complex_dtype)
        L_batch_size = L if L_batch_size is None else L_batch_size
        for l1 in tqdm(range(0, L, L_batch_size), 'Least Squares Forward Pass'):
            l2 = min(l1 + L_batch_size, L)
            AHy[l1:l2, ...] = st_imperf.forward_matrix_prod(spatial_funcs[l1:l2].conj()) / grd.shape[1]

        # Solve for h_l(t) using least squares
        AHA_inv = torch.linalg.pinv(AHA)
        temporal_funcs = einsum(AHA_inv, AHy, 'Lo Li, Li ... -> Lo ...')

    return spatial_funcs, temporal_funcs