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

def svd_decomp_fast_temporal(st_imperf: spatio_temporal,
                             L: int) -> torch.Tensor:
    """
    Perform SVD decomposition of the spatio temporal imperfection,
    specifically with fast temporal operator axis.

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
    U, S, Vh = svd_power_method_tall(A=st_imperf.adjoint_matrix_prod, 
                                     AHA=st_imperf.adjoint_forward_matrix_prod,
                                     inp_dims=st_imperf.trj_size,
                                     rank=L,
                                     device=st_imperf.torch_dev)
    spatial_funcs = rearrange(U * S, '... nseg -> nseg ...').conj()
    temporal_funcs = Vh.conj()

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
        temp_features = torch.tile(clusters[i], (grd.shape[1], 1)).T
        spatial_funcs[i, ...] = st_imperf.matrix_access(grd, temp_features).reshape(im_size)

    # Compute temporal basis functions
    temporal_funcs = torch.zeros(L, *st_imperf.trj_size, device=st_imperf.torch_dev)
    if 'zero' in interp_type:
        for i in range(L):
            temporal_funcs[i, ...] = (inds == i) * 1.0
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