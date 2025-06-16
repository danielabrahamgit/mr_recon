"""
Implimentation of:
Region-optimized virtual (ROVir) coils: Localization and/orsuppression of spatial regions using sensor-domain beamforming
https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.28706
"""

import torch

from scipy.linalg import eig, eigh
from einops import rearrange, einsum
from logging import warning
from typing import Optional, Union
from mr_recon.utils import np_to_torch

def apply_rovir(img_cal: torch.Tensor,
                mask_signal: torch.Tensor,
                mask_interf: torch.Tensor,
                signal_cutoff: Optional[Union[float, int]] = 0.95,
                B_eps: Optional[float] = 1e-6,
                *channel_data) -> torch.Tensor:
    """
    Apply the ROvir algorithm to the input image.
    
    Parameters
    ----------
    img_cal : torch.Tensor
        Calibration image of shape (C, *im_size).
    mask_signal : torch.Tensor
        Mask of the signal region of shape (*im_size).
    mask_interf : torch.Tensor
        Mask of the interference region of shape (*im_size).
    signal_cutoff : float or int
        if float, percent SVD energy to keep
        if int, number of SVD components to keep
    B_eps : float
        B = B + B_eps * I if B is singular.
    channel_data 
        Additional channel data to apply rovir weights to
    
    Returns
    -------
    W : torch.Tensor
        output weights of shape (C, n), n is smaller than C
    ch_data : list (only if channel_data is provided)
        Compressed channel data
    """
    # Consts
    C = img_cal.shape[0]
    
    # Compute A and B matrices
    A, B = compute_AB_matrices(img_cal, mask_signal, mask_interf)
    
    # Add small value to B to avoid singularity
    eig_b = torch.linalg.eigvalsh(B)
    if eig_b.min().item() < 0:
        B_psd = B - torch.eye(C, device=B.device, dtype=B.dtype) * eig_b.min()
    else:
        B_psd = B.clone()
    B_psd += B_eps * torch.eye(C, device=B.device, dtype=B.dtype) * eig_b.max()
    
    # # Compute generalized eigenvalue decomposition
    evals, evecs = generalized_eigenvalue_decomposition(A, B_psd)
    
    # Determine cutoff number of channels
    pcnt_signal = torch.zeros(C, device=img_cal.device)
    pcnt_interf = torch.zeros(C, device=img_cal.device)
    for k in range(1, C+1):
        W = evecs[:, :k]
        W = torch.linalg.qr(W, mode='reduced')[0] # C k
        img_cal_rovir = einsum(img_cal, W, 'nc ..., nc nc2 -> nc2 ...')
        pcnt_signal[k-1] = (img_cal_rovir * mask_signal).norm() / (img_cal * mask_signal).norm()
        pcnt_interf[k-1] = (img_cal_rovir * mask_interf).norm() / (img_cal * mask_interf).norm()
    
    # Apply cutoff
    if type(signal_cutoff) == float:
        assert pcnt_signal.max() >= signal_cutoff, "signal_cutoff is too large"
        signal_cutoff = (pcnt_signal >= signal_cutoff).nonzero(as_tuple=True)[0][0] + 1
        signal_cutoff = signal_cutoff.item()
    signal_cutoff = min(signal_cutoff, C)
    weights = evecs[:, :signal_cutoff] # C n
    weights = torch.linalg.qr(weights, mode='reduced')[0]
    
    import matplotlib.pyplot as plt
    plt.plot(pcnt_signal.cpu().numpy(), label='Signal')
    plt.plot(pcnt_interf.cpu().numpy(), label='Interference')
    plt.axvline(signal_cutoff, color='r', linestyle='--')
    plt.ylim(-.05, 1.05)
    plt.legend()
    
    # Show the percent of signal and interference energy
    print(f'Signal pcnt = {100*pcnt_signal[signal_cutoff-1].item():.2f}%')
    print(f'Interf pcnt = {100*pcnt_interf[signal_cutoff-1].item():.2f}%')
    
    # Apply weights to channel data
    ch_data = []
    for arg in channel_data:
        arg_torch = np_to_torch(arg)
        arg_compressed = einsum(arg_torch, weights, 'nc ..., nc nc2 -> nc2 ...')
        ch_data.append(arg_compressed)
        
    # Return the compressed channel data or just the weights
    if len(ch_data) == 0:
        return weights
    else:
        return [weights] + ch_data

def compute_AB_matrices(img_cal: torch.Tensor,
                        mask_signal: torch.Tensor,
                        mask_interf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the A and B matrices for the ROvir algorithm.
    
    Parameters
    ----------
    img_cal : torch.Tensor
        Calibration image of shape (C, *im_size).
    mask_signal : torch.Tensor
        Mask of the signal region of shape (*im_size).
    mask_interf : torch.Tensor
        Mask of the interference region of shape (*im_size).
    B_eps : float
        B = B + B_eps * I if B is singular.
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A matrix of shape (C, C) and B matrix of shape (C, C).
    """
    # Consts
    C = img_cal.shape[0]
    
    # Compute A as signal covariance matrix
    cal_sig = (img_cal * mask_signal).reshape((C, -1)).T
    A = cal_sig.H @ cal_sig
    
    # Compute B as interference covariance matrix
    cal_interf = (img_cal * mask_interf).reshape((C, -1)).T # N C
    B = cal_interf.H @ cal_interf
    
    return A, B

def generalized_eigenvalue_decomposition(A: torch.Tensor,
                                         B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the generalized eigenvalue decomposition of the A and B matrices.
    
    Parameters
    ----------
    A : torch.Tensor
        A matrix of shape (N, N).
    B : torch.Tensor
        B matrix of shape (N, N).
    
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Eigenvalues and eigenvectors.
    """
    # Consts
    N = A.shape[0]
    assert A.shape == (N, N), "A must be square"
    assert B.shape == (N, N), "B must be square"
    if A.device != A.cpu().device:
        warning('This is not GPU accelerated.')
        
    # Compute the generalized eigenvalue decomposition using scipy
    evals, evecs = eigh(A.cpu().numpy(), B.cpu().numpy())
    
    # Back to torch
    evals = np_to_torch(evals).to(A.device).abs()
    evecs = np_to_torch(evecs).to(A.device)
    
    # Sort
    inds = evals.argsort(descending=True)
    evals = evals[inds]
    evecs = evecs[:, inds]
    
    return evals, evecs