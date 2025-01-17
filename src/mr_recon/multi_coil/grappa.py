import torch

from typing import Optional
from tqdm import tqdm
from einops import rearrange, einsum

def non_cartesian_GRAPPA_adjoint(kernel_model: callable,
                                 ksp_trg: torch.Tensor,
                                 trj_acq: torch.Tensor,
                                 trj_trg: torch.Tensor,
                                 src_inds: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a trained kernel model, this function applies the adjoint GRAPPA kernel to the target non-cartesian k-space data.
    
    Parameters:
    ----------
    kernel_model : callable
        GRAPPA kernel model that takes in source vectors and source data, and outputs the target data and kernel.
    ksp_trg : torch.Tensor
        Acquired k-space data with shape (C, N)
    trj_acq : torch.Tensor
        Acquired trajectory data with shape (M, d)
    trj_trg : torch.Tensor
        Target trajectory data with shape (N, d)
    src_inds : torch.Tensor
        Source indices with shape (N, S) with values in the range [0, M)
        
    Returns:
    --------
    ksp_acq : torch.Tensor
        adjoint output 'acquired' k-space data with shape (C, M)
    kerns : torch.Tensor
        GRAPPA kernel with shape (N, kern_size)
    """
    # Build source vectors
    trj_src = trj_acq[src_inds] # N S d
    src_vecs = trj_src - trj_trg[:, None] # N S d
    
    # Apply kernel model
    M, d = trj_acq.shape
    C, N = ksp_trg.shape
    N, S = src_inds.shape
    ksp_src_fake = torch.zeros((N, S, C), device=ksp_trg.device, dtype=ksp_trg.dtype)
    kerns = kernel_model.forward(rearrange(src_vecs, 'N S d -> N (S d)'), 
                                 ksp_src_fake)[1]
    kerns = rearrange(kerns, 'N C (S Ci) -> N C Ci S', Ci=C).conj()
    
    # Apply adjoint kernel
    ksp_acq = torch.zeros((C, M), device=ksp_trg.device, dtype=ksp_trg.dtype)
    kerns_weighted = einsum(kerns, ksp_trg, 'N C Ci S, C N -> Ci N S').reshape((C, -1))
    ksp_acq.index_add_(1, src_inds.flatten(), kerns_weighted)
    
    return ksp_acq, kerns

def non_cartesian_GRAPPA(kernel_model: callable,
                         ksp_acq: torch.Tensor,
                         trj_acq: torch.Tensor,
                         trj_trg: torch.Tensor,
                         src_inds: torch.Tensor,
                         feature_vecs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a trained kernel model, this function applies the GRAPPA kernel to the acquired non-cartesian k-space data.
    
    Parameters:
    ----------
    kernel_model : callable
        GRAPPA kernel model that takes in source vectors and source data, and outputs the target data and kernel.
    ksp_acq : torch.Tensor
        Acquired k-space data with shape (C, M)
    trj_acq : torch.Tensor
        Acquired trajectory data with shape (M, d)
    trj_trg : torch.Tensor
        Target trajectory data with shape (N, d)
    src_inds : torch.Tensor
        Source indices with shape (N, S) with values in the range [0, M)
    feature_vecs : Optional[torch.Tensor]
        Source feature vectors with shape (N, f)
        if None, then the source vectors are used as feature vectors
        
    Returns:
    --------
    ksp_trg : torch.Tensor
        Target k-space data with shape (C, N)
    kerns : torch.Tensor
        GRAPPA kernel with shape (N, kern_size)
    """
    # Build source vectors
    trj_src = trj_acq[src_inds] # N S d
    src_vecs = trj_src - trj_trg[:, None] # N S d
    
    # Feature vector default
    if feature_vecs is None:
        feature_vecs = rearrange(src_vecs, 'N S d -> N (S d)')
    
    # Grab source samples
    ksp_src = ksp_acq[:, src_inds] # C N S
    
    # Apply kernel model
    ksp_trg, kerns = kernel_model.forward(feature_vecs,
                                          rearrange(ksp_src, 'C N S -> N S C'))
    ksp_trg = rearrange(ksp_trg, 'N C -> C N')
    
    return ksp_trg, kerns