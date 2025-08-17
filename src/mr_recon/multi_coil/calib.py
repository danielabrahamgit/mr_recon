import torch
import numpy as np

from mr_recon.fourier import fft
from mr_recon.utils import np_to_torch, torch_to_np
from mr_recon.linops import multi_chan_linop
from mr_recon.recons import CG_SENSE_recon

from einops import rearrange, einsum
from typing import Optional, Union

def whiten_data(noise_mat: torch.Tensor,
                *args):
    """
    Whiten data using calibration noisy data

    Args:
    -----
    noise_mat : torch.Tensor
        matrix with just noise, shape (C, ...)
    *args : list[torch.Tensor]
        additional tensors to whiten, shape (C, ...)

    Returns:
    --------
    psi_half_inv : torch.Tensor
        whitening matrix, shape (C, C)
    args : list[torch.Tensor]
        whitened tensors, shape (C, ...)
    """
    C = noise_mat.shape[0]
    noise_mat = noise_mat.reshape((C, -1))
    noise_mat -= noise_mat.mean(dim=-1, keepdim=True)
    cov_mat = (noise_mat @ noise_mat.H)
    cov_mat /= noise_mat.abs().max()
    vals, vecs = torch.linalg.eigh(cov_mat)
    psi_half_inv = (vecs * (vals ** (-0.5))) @ vecs.H
    
    whitened = []
    for arg in args:
        arg_whitened = einsum(arg, psi_half_inv, 'C ..., Co C -> Co ...')
        whitened.append(arg_whitened)

    if len(whitened) == 0:
        return psi_half_inv

    else:
        return [psi_half_inv] + whitened

def calc_coil_subspace(ksp_mat: torch.Tensor,
                       new_coil_size: Union[int, float],
                       *args):

    # Move to torch
    np_flag = type(ksp_mat) is np.ndarray
    ksp_mat = np_to_torch(ksp_mat)

    # Estimate coil subspace
    n_coil = ksp_mat.shape[0]
    u, s, vt = torch.linalg.svd(ksp_mat.reshape((n_coil, -1)), full_matrices=False)
    if new_coil_size is None:
        new_coil_size = n_coil
    elif type(new_coil_size) is float:
        cmsm = torch.cumsum(s, dim=0)
        n_coil = torch.argwhere(cmsm > new_coil_size * cmsm[-1]).squeeze()[0]
    elif type(new_coil_size) is int:
        n_coil = new_coil_size
    coil_subspace = u[:, :n_coil] #/ s[:n_coil]
    coil_subspace.imag *= -1 # conj without .conj() for weird numpy conversion reasons

    comps = []
    for arg in args:
        arg_torch = np_to_torch(arg)
        arg_compressed = einsum(arg_torch, coil_subspace, 'nc ..., nc nc2 -> nc2 ...')
        if np_flag:
            arg_compressed = torch_to_np(arg_compressed)
        comps.append(arg_compressed)

    if len(comps) == 0:
        return coil_subspace

    else:
        return [coil_subspace] + comps

def synth_cal(ksp: torch.Tensor, 
              cal_size: Union[tuple, int],
              trj: Optional[torch.Tensor] = None,
              dcf: Optional[torch.Tensor] = None,
              num_iter: Optional[int] = 0,
              use_toeplitz: Optional[bool] = False) -> torch.Tensor:
    """
    Synthesizes a calibration region from data.
    
    Parameters:
    -----------
    ksp: torch.Tensor
        k-space raw data with shape (C *trj_size)
    cal_size: tuple or int
        width of calibration along each dimension (isotropic if int)
    trj: torch.Tensor
        trajectory with shape (*trj_size D)
    dcf: torch.Tensor
        density compensation function with shape (*trj_size)
    num_iter: int
        number of conj gradient iterations for estimating rectilinear calib from non-cart data
    use_toeplitz: bool
        use toeplitz nufft for estimating rectilinear calib from non-cart data
    
    Returns:
    --------
    ksp_cal: torch.Tensor
        calibration signal with shape (C, *cal_size)
    """
    # Consts
    C = ksp.shape[0]
    if type(cal_size) == int:
        cal_size = (cal_size,) * D
        
    # Handle cartesian case first, assume calib is at center of k-space matrix
    if trj is None:
        D = ksp.ndim - 1
        tup = tuple([slice(ksp.shape[i+1]//2 - cal_size[i]//2, ksp.shape[i+1]//2 + cal_size[i]//2) for i in range(D)])
        tup = (slice(None),) + tup
        ksp_cal = ksp[tup]
    # Non-Cartesian case, do CG to estimate rectilinear calib
    else:
        # Truncate trajectory and k-space
        D = trj.shape[-1]
        inds = tuple(torch.argwhere(torch.all(trj.abs() < torch.tensor(cal_size, device=trj.device) / 2, dim=-1)).T)
        inds_ksp = (slice(None),) + inds
        trj_cut = trj[inds]
        ksp_cut = ksp[inds_ksp]
        if dcf is not None:
            dcf_cut = dcf[inds]
        else:
            dcf_cut = None
        
        # Recon
        A = multi_chan_linop((C, *cal_size), trj_cut, dcf_cut, use_toeplitz=use_toeplitz)
        img_cal = CG_SENSE_recon(A, ksp_cut, max_iter=num_iter, lamda_l2=0.0, max_eigen=1.0, verbose=False)
        ksp_cal = fft(img_cal, dim=tuple(range(-D, 0)))
    
    return ksp_cal
        