from sympy import adjoint
import torch

from typing import Optional
from mr_recon.imperfections.spatio_temporal_imperf import spatio_temporal
from mr_recon.algs import svd_power_method_tall
from einops import rearrange, einsum
from mr_recon.utils import gen_grd

"""
All of these techniques are used to decompose a spatio-temporal imperfection:
    
    W(r, p(t)) = sum_{l=1}^{L} h_l(t) b_l(r)
"""
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
    temporal_features = temporal_features[:, None, :] # ntrj 1 f
    mat = st_imperf.matrix_access(spatial_inds, temporal_features)
    
    # SVD
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    temporal_funcs = (U[:, :L] * (S[None, :L] ** 0.5)).T.reshape((L, *trj_size))
    spatial_funcs = (Vh[:L, :] * (S[:L, None] ** 0.5)).reshape((L, *im_size))

    return spatial_funcs, temporal_funcs

def svd_decomp_operator(st_imperf: spatio_temporal,
                        L: int,
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
    if nvox < ntrj:
        fast_axis = 'spatial'
    else:
        fast_axis = 'temporal'

    if 'temporal' in fast_axis:
        U, S, Vh = svd_power_method_tall(A=st_imperf.adjoint_matrix_prod, 
                                        AHA=st_imperf.adjoint_forward_matrix_prod,
                                        inp_dims=st_imperf.trj_size,
                                        rank=L,
                                        device=st_imperf.torch_dev)
        spatial_funcs = rearrange(U * (S ** 0.5), '... L -> L ...').conj()
        temporal_funcs = einsum(Vh, (S ** 0.5), 'L ..., L -> L ...').conj()
    elif 'spatial' in fast_axis:
        U, S, Vh = svd_power_method_tall(A=st_imperf.forward_matrix_prod, 
                                        AHA=st_imperf.forward_adjoint_matrix_prod,
                                        inp_dims=st_imperf.im_size,
                                        rank=L,
                                        device=st_imperf.torch_dev)
        temporal_funcs = rearrange(U * (S ** 0.5), '... L -> L ...')
        spatial_funcs = einsum(Vh, (S ** 0.5), 'L ..., L -> L ...')


    return spatial_funcs, temporal_funcs

def svd_decomp_operator_coils(st_imperf: spatio_temporal,
                              mps: torch.Tensor,
                              L: int,
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
    ncoil = mps.shape[0]
    if nvox < ntrj:
        fast_axis = 'spatial'
    else:
        fast_axis = 'temporal'

    if 'temporal' in fast_axis:
        A = lambda x: st_imperf.adjoint_matrix_prod(x)
        AHA = lambda x: st_imperf.adjoint_forward_matrix_prod(x)
        U, S, Vh = svd_power_method_tall(A=A, 
                                        AHA=AHA,
                                        inp_dims=(ncoil, *st_imperf.trj_size),
                                        rank=L,
                                        device=st_imperf.torch_dev)
        spatial_funcs = rearrange(U * (S ** 0.5), '... L -> L ...').conj()
        temporal_funcs = einsum(Vh, (S ** 0.5), 'L ..., L -> L ...').conj()
    elif 'spatial' in fast_axis:
        A = lambda x: st_imperf.forward_matrix_prod(x)
        AHA = lambda x: st_imperf.adjoint_forward_matrix_prod(x)
        U, S, Vh = svd_power_method_tall(A=A,
                                        AHA=AHA,
                                        inp_dims=st_imperf.im_size,
                                        rank=L,
                                        device=st_imperf.torch_dev)
        temporal_funcs = rearrange(U * (S ** 0.5), '... L -> L ...')
        spatial_funcs = einsum(Vh, (S ** 0.5), 'L ..., L -> L ...')

    return spatial_funcs, temporal_funcs

def temporal_segmentation(st_imperf: spatio_temporal,
                          L: int,
                          interp_type: Optional[str] = 'zero') -> torch.Tensor:
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
    spatial_funcs = torch.zeros(L, *im_size, device=st_imperf.torch_dev, dtype=torch.complex64)
    grd = gen_grd(im_size, im_size).reshape((-1, len(im_size)))
    grd = (grd + torch.tensor(im_size)//2).type(torch.int).T
    grd = grd.to(st_imperf.torch_dev)
    for i in range(L):
        temp_features = torch.tile(clusters[i, None, :], (grd.shape[1], 1))
        spatial_funcs[i, ...] = st_imperf.matrix_access(grd, temp_features).reshape(im_size)

    # Compute temporal basis functions
    temporal_funcs = torch.zeros(L, *st_imperf.trj_size, device=st_imperf.torch_dev, dtype=torch.complex64)
    if 'zero' in interp_type:
        for i in range(L):
            temporal_funcs[i, ...] = ((inds == i) * 1.0)
    elif 'linear' in interp_type:
        raise NotImplementedError
    elif 'lstsq' in interp_type:
        # First compute AHA, which is all the pairwise 
        # dot products of the spatial features
        AHA = torch.zeros((L, L), device=st_imperf.torch_dev, dtype=torch.complex64)
        for i in range(L):
            for j in range(L):
                AHA[i, j] = (spatial_funcs[i].conj() * spatial_funcs[j]).mean()

        # Next compute AHy, which is essentially int_r W(r, t) * b_l(r).conj() dr
        AHy = torch.zeros((L, *st_imperf.trj_size), device=st_imperf.torch_dev, dtype=torch.complex64)
        for i in range(L):
            AHy[i, ...] = st_imperf.forward_matrix_prod(spatial_funcs[i].conj()) / grd.shape[1]

        # Solve for h_l(t) using least squares
        AHA_inv = torch.linalg.pinv(AHA)
        temporal_funcs = einsum(AHA_inv, AHy, 'Lo Li, Li ... -> Lo ...')

    return spatial_funcs, temporal_funcs