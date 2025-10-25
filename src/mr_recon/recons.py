import gc
import torch
import time

from tqdm import tqdm
from typing import Optional, Union
from mr_recon.dtypes import complex_dtype
from mr_recon.linops import linop
from mr_recon.utils import np_to_torch, torch_to_np
from mr_recon.algs import (
    density_compensation, 
    conjugate_gradient, 
    power_method_operator, 
    gradient_descent,
    FISTA,
    TOS,
)

def _max_eigen_calc(A: linop,
                    device: torch.device,
                    verbose: Optional[bool] = True,
                    max_eigen: Optional[Union[float, str]] = 'rough') -> float:
    """
    Estimate largest eigenvalue so that lambda max of AHA is 1
    """
    if max_eigen is None or max_eigen == 'rough':
        x0 = torch.randn(A.ishape, dtype=complex_dtype, device=device)
        if max_eigen == 'rough':
            # faster max eigenvalue that is less accurate but good enough
            _, max_eigen = power_method_operator(A.normal, x0, num_iter=6, verbose=verbose)
            max_eigen *= 1.3
        else:
            _, max_eigen = power_method_operator(A.normal, x0, num_iter=12, verbose=verbose)
            max_eigen *= 1.1
    else:
        assert isinstance(max_eigen, (float, int, torch.Tensor)), "max_eigen must be a tensor, float, 'rough', or None"
    
    return max_eigen

def prep_AHA_AHb(A: linop,
                 ksp: torch.Tensor,
                 max_eigen: Optional[Union[float, str]] = 'rough',
                 ahb_init: Optional[torch.Tensor] = None,
                 verbose: bool = True,
                 alpha_scale: bool = True,
                 ) -> tuple[callable, torch.Tensor]:

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    max_eigen = _max_eigen_calc(A, device, verbose, max_eigen)
    
    # Starting with AHb
    if ahb_init is None:
        start = time.perf_counter()
        y = ksp.type(complex_dtype) / (max_eigen ** 0.5)

        if alpha_scale:
            ynorm = y.abs().square().sum()

        AHb = A.adjoint(y) / (max_eigen ** 0.5)
        end = time.perf_counter()
        del y
        if verbose:
            print(f'AHb took {end-start:.3f}(s)')
    else:
        alpha_scale = False
        AHb = ahb_init.type(complex_dtype) / (max_eigen)
    
    # Wrap normal with max eigen
    AHA = lambda x : A.normal(x) / max_eigen

    if alpha_scale:
        # NOTE: try Adding scaling to improve conditioning
        scale = ynorm / ((A(AHb) / (max_eigen ** 0.5)).abs().square().sum())
        AHb = AHb * scale
        if verbose:
            print(f"\tAlpha Scale = {scale:.4f}")
    else:
        scale = 1.0

    return AHA, AHb, scale

def min_norm_recon(A: linop,
                   ksp: torch.Tensor,
                   max_iter: int = 15,
                   lamda_l2: Optional[float] = 0.0,
                   max_eigen: Optional[Union[float, str]] = 'rough',
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
    lamda_l2 : float
        l2 lamda regularization
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
    max_eigen = _max_eigen_calc(A, device, verbose, max_eigen)

    # Wrap normal with max eigen
    AAH = lambda x : A.forward(A.adjoint(x)) / max_eigen

    # Run CG
    y = conjugate_gradient(AHA=AAH, 
                           AHb=ksp.type(complex_dtype),
                           lamda_l2=lamda_l2,
                           num_iters=max_iter,
                           verbose=verbose)
    
    # Apply adjoint 
    start = time.perf_counter()
    recon = A.adjoint(y) / (max_eigen)
    end = time.perf_counter()
    if verbose:
        print(f'AHy took {end-start:.3f}(s)')
    
    return recon

def CG_SENSE_recon(A: linop,
                   ksp: torch.Tensor,
                   max_iter: Optional[int] = 15,
                   lamda_l2: Optional[float] = 0.0,
                   max_eigen: Optional[Union[float, str]] = 'rough',
                   tolerance: Optional[float] = 1e-8,
                   weights: Optional[torch.Tensor] = None,
                   ahb_init: Optional[torch.Tensor] = None,
                   clear_gpu_mem: Optional[bool] = True,
                   verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run CG SENSE recon:
    recon = (AHA + lamda_l2I)^-1 AHb
    
    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, ...)
    max_iter : int
        max number of iterations for recon algorithm
    lamda_l2 : float
        l2 lamda regularization for SENSE: ||Ax - b||_2^2 + lamda_l2||x||_2^2
    max_eigen : float
        maximum eigenvalue of AHA
    tolerance : float
        tolerance for CG algorithm
    ahb_init : torch.Tensor
        initial value for AHb, if None, it will be computed
    verbose : bool 
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """
    
    AHA, AHb, scale = prep_AHA_AHb(A, ksp, max_eigen, ahb_init, verbose)
    
    if max_iter == 0:
        return AHb

    if clear_gpu_mem:
        gc.collect()
        with ksp.device:
            torch.cuda.empty_cache()

    # Run CG
    recon = conjugate_gradient(AHA=AHA, 
                               AHb=AHb,
                               num_iters=max_iter,
                               lamda_l2=lamda_l2,
                               tolerance=tolerance,
                               weights=weights,
                               verbose=verbose)
    
    return recon / scale

def coil_combine(multi_chan_img: torch.Tensor,
                 mps: Optional[torch.Tensor] = None,
                 walsh_kernel_size: Optional[int] = None) -> torch.Tensor:
    """
    Combine multi-channel images using SENSE, walsh, or SoS

    Parameters:
    -----------
    multi_chan_img : torch.Tensor
        multi-channel image with shape (nc, ...)
    mps : torch.Tensor
        coil sensitivity maps with shape (nc, ...)
    walsh_kernel_size : int
        size of walsh kernel for walsh coil combination
    
    Returns:
    --------
    img_comb : torch.Tensor
        the combined image/volume with shape (...)
    """

    if mps is not None:
        img_comb = (multi_chan_img * mps.conj()).sum(0) / (mps.abs().square().sum(0) + 1e-5)
    elif walsh_kernel_size is not None:
        # Reshape image into blocks 
        raise NotImplementedError
    else:
        img_comb = multi_chan_img.abs().square().sum(0).sqrt()
    
    return img_comb

def FISTA_recon(A: linop,
                ksp: torch.Tensor,
                proxg: callable,
                max_iter: int = 40,
                max_eigen: Optional[Union[float, str]] = 'rough',
                ahb_init: Optional[torch.Tensor] = None,
                clear_gpu_mem: Optional[bool] = True,
                verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run FISTA recon
    recon = min_x ||Ax - b||_2^2 + g(x)
    
    Parameters
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

    Returns
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    AHA, AHb, scale = prep_AHA_AHb(A, ksp, max_eigen, ahb_init, verbose)
    
    if max_iter == 0:
        return AHb

    if clear_gpu_mem:
        gc.collect()
        with ksp.device:
            torch.cuda.empty_cache()

    # Run FISTA
    recon = FISTA(AHA, AHb, proxg, max_iter, verbose=verbose)

    return recon / scale

def TOS_recon(A: linop,
              ksp: torch.Tensor,
              prox1: callable,
              prox2: callable,
              max_iter: int = 50,
              max_eigen: Optional[Union[float, str]] = 'rough',
              ahb_init: Optional[torch.Tensor] = None,
              clear_gpu_mem: Optional[bool] = True,
              verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Run TOS recon
    recon = min_x ||Ax - b||_2^2 + g1(x) + g2(x)
    
    Parameters
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

    Returns
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    AHA, AHb, scale = prep_AHA_AHb(A, ksp, max_eigen, ahb_init, verbose)
    
    if max_iter == 0:
        return AHb

    if clear_gpu_mem:
        gc.collect()
        with ksp.device:
            torch.cuda.empty_cache()

    # Run FISTA
    recon = TOS(AHA, AHb, prox1, prox2, num_iters=max_iter, verbose=verbose)

    return recon / scale