import time
import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Optional

def density_compensation(trj: torch.Tensor,
                         im_size: tuple,
                         num_iters: Optional[int] = 30):
    raise NotImplementedError

def largest_eigenvalue(A: nn.Module,
                       x0: torch.Tensor,
                       num_iters: Optional[int] = 15,
                       verbose: Optional[bool] = True) -> float:
    """
    Estimates the largest eigenvalue of A.

    Parameters
    ----------
    A : nn.Module
        Linear operator, input shape and output shape should be the same
    x0 : torch.tensor
        Initial guess of largest eigenvector
    num_iters : int
        Number of iterations
    verbose : bool
        toggles print statements

    Returns
    ---------
    lamda_max : float
        Largest eigenvalue of A
    """
    
    for _ in tqdm(range(num_iters), 'Max Eigenvalue', disable=not verbose):
        
        z = A(x0)
        ll = torch.norm(z)
        x0 = z / ll
    
    if verbose:
        print(f'Max Eigenvalue = {ll}')
    
    return ll.item()

def FISTA(AHA: nn.Module, 
          AHb: torch.Tensor, 
          proxg: nn.Module, 
          num_iters: Optional[int] = 20,
          ptol_exit: Optional[float] = 0.5,
          verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Solves ||Ax - b||_2^2 + lamda ||Gx||_1, where G is a linear function.
    The proximal operator of Gx if given by 'proxg'

    Parameters
    ----------
    AHA : nn.Module
        The gram or normal operator of A
    AHb : torch.tensor
        The A hermitian transpose times b
    proxg : nn.Module
        The torch
    num_iters : int
        Number of iterations
    ptol_exit : float
        percent tolerance exit condition
    verbose : bool
        toggles print statements

    Returns
    ---------
    x : torch.tensor
        Reconstructed tensor
    """

    # Start at AHb
    x = AHb.clone()
    z = x.clone()
    
    if num_iters <= 0:
        return x

    # Benchmarking
    t_gr = 0.0
    t_prox = 0.0

    for k in tqdm(range(0, num_iters), 'FISTA Iterations', disable=not verbose):

        x_old = x.clone()
        x     = z.clone()
        
        start = time.perf_counter()
        gr    = AHA(x) - AHb
        end = time.perf_counter()
        t_gr += end - start
        start = time.perf_counter()
        x     = proxg(x - gr)
        end = time.perf_counter()
        t_prox += end - start
        if k == 0:
            z = x
        else:
            step  = k/(k + 3)
            z     = x + step * (x - x_old)
        ptol = 100 * torch.norm(x_old - x)/torch.norm(x)
        if ptol < ptol_exit:
            if verbose:
                print(f'Tolerance reached after {k+1} iterations, exiting FISTA')
            break

    if verbose:
        print(f'Gradient Took {t_gr:.3f}(s), Prox Took {t_prox:.3f}(s)')
    return x

def conjugate_gradient(AHA: nn.Module, 
                       AHb: torch.Tensor, 
                       P: Optional[nn.Module] = None,
                       num_iters: Optional[int] = 10, 
                       lamda_l2: Optional[float] = 1e-2,
                       tolerance: Optional[float] = 1e-8,
                       verbose=True) -> torch.Tensor:
    """Conjugate gradient for complex numbers. The output is also complex.
    Solve for argmin ||Ax - b||^2. Inspired by sigpy!
    
    Parameters:
    -----------
    AHA : nn.Module 
        Linear operator representing the gram/normal operator of A
    AHb : torch.tensor
        The A hermitian transpose times b
    P : nn.Module
        Preconditioner 
    num_iters : int 
        Max number of iterations.
    lamda_l2 : float
        Replaces AHA with AHA + lamda_l2 * I
    tolerance : float 
        Used as stopping criteria
    verbose : bool
        toggles print statements
    
    Returns:
    ---------
    x : torch.tensor <complex>
        least squares estimate of x, same shape as x0 if provided    
    """

    # Default preconditioner is identity matrix
    if P is None:
        P = lambda x : x
    
    # Tikonov regularization
    AHA_wrapper = lambda x : AHA(x) + lamda_l2 * x

    # Start at AHb
    x0 = AHb.clone()

    # Define iterative vars
    r = AHb - AHA_wrapper(x0)

    z = P(r)
    p = z.clone()

    best_sol = x0.clone()
    best_res = torch.norm(r)
    best_iter = -1

    if torch.norm(r) < tolerance or num_iters == 0:
        print(f'Best Residual = {best_res}, Best Iteration = {best_iter}')
        return x0

    # Main loop
    for i in tqdm(range(num_iters), 'CG Iterations', disable=not verbose):

        # Apply model
        Ap = AHA_wrapper(p)
        pAp = torch.real(torch.sum(p.conj() * Ap)).item()

        # Update x
        # assert pAp > 0, 'A is not Semi-Definite'
        rz = torch.real(torch.sum(r.conj() * z))
        alpha = rz / pAp
        x0 = x0 + alpha * p

        # Update r
        r = r - alpha * Ap
        if torch.norm(r) < best_res:
            best_iter = i
            best_sol = x0.clone()
            best_res = torch.norm(r)

        if torch.norm(r) < tolerance:
            break

        # Update z
        z = P(r)

        # Update p
        beta = torch.real(torch.sum(r.conj() * z)) / rz
        p = z + beta * p

    print(f'Best Residual = {best_res}, Best Iteration = {best_iter}')
    return best_sol