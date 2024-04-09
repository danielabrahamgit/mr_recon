import os
import torch
import time
import numpy as np
import sigpy as sp

from tqdm import tqdm
from typing import Optional
from mr_recon.linops import linop
from mr_recon.utils import np_to_torch, torch_to_np
from mr_recon.algs import (
    density_compensation, 
    conjugate_gradient, 
    power_method_operator, 
    FISTA
)

def min_norm_recon(A: linop,
                   ksp: torch.Tensor,
                   max_iter: int = 15,
                   max_eigen: Optional[float] = None,
                   verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run min norm recon:
    recon = A^H(AA^H)^-1b
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, ...)
    max_iter : int
        max number of iterations for recon algorithm
    max_eigen : float
        maximum eigenvalue of AHA
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=torch.complex64, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01

    # Wrap normal with max eigen
    AAH = lambda x : A.forward(A.adjoint(x)) / max_eigen

    # Run CG
    y = conjugate_gradient(AHA=AAH, 
                           AHb=ksp.type(torch.complex64),
                           num_iters=max_iter,
                           verbose=verbose)
    
    # Apply adjoint 
    start = time.perf_counter()
    recon = A.adjoint(y) / (max_eigen ** 0.5)
    end = time.perf_counter()
    if verbose:
        print(f'AHy took {end-start:.3f}(s)')
    
    return recon

def CG_SENSE_recon(A: linop,
                   ksp: torch.Tensor,
                   max_iter: int = 15,
                   lamda_l2: float = 0.0,
                   max_eigen: Optional[float] = None,
                   verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run CG SENSE recon:
    recon = (AHA + lamda_l2I)^-1 AHb
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, nro, npe, ntr)
    max_iter : int
        max number of iterations for recon algorithm
    lamda_l2 : float
        l2 lamda regularization for SENSE: ||Ax - b||_2^2 + lamda_l2||x||_2^2
    max_eigen : float
        maximum eigenvalue of AHA
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=torch.complex64, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01
    
    # Starting with AHb
    start = time.perf_counter()
    y = ksp.type(torch.complex64)
    AHb = A.adjoint(y) / (max_eigen ** 0.5)
    end = time.perf_counter()
    if verbose:
        print(f'AHb took {end-start:.3f}(s)')

    # Clear data (we dont need it anymore)
    y = y.cpu()
    with device:
        torch.cuda.empty_cache()

    # Wrap normal with max eigen
    AHA = lambda x : A.normal(x) / max_eigen

    # Run CG
    recon = conjugate_gradient(AHA=AHA, 
                               AHb=AHb,
                               num_iters=max_iter,
                               lamda_l2=lamda_l2,
                               verbose=verbose)
    
    return recon

def FISTA_recon(A: linop,
                ksp: torch.Tensor,
                proxg: callable,
                max_iter: int = 40,
                max_eigen: Optional[float] = None,
                verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run FISTA recon
    recon = min_x ||Ax - b||_2^2 + g(x)
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, nro, npe, ntr)
    proxg : callable
        proximal operator for g(x)
    max_iter : int
        max number of iterations for recon algorithm
    max_eigen : float
        maximum eigenvalue of AHA
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=torch.complex64, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01
    
    # Starting with AHb
    start = time.perf_counter()
    y = ksp.type(torch.complex64)
    AHb = A.adjoint(y) / (max_eigen ** 0.5)
    end = time.perf_counter()
    if verbose:
        print(f'AHb took {end-start:.3f}(s)')

    # Clear data (we dont need it anymore)
    y = y.cpu()
    with device:
        torch.cuda.empty_cache()

    # Wrap normal with max eigen
    AHA = lambda x : A.normal(x) / max_eigen

    # Run FISTA
    recon = FISTA(AHA, AHb, proxg, max_iter)

    return recon
