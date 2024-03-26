import time
import torch
import sigpy as sp
import torch.nn as nn

from tqdm import tqdm
from typing import Optional, Tuple
from einops import rearrange
from mr_recon.utils import torch_to_np, np_to_torch

def density_compensation(trj: torch.Tensor,
                         im_size: tuple,
                         num_iters: Optional[int] = 30):
    raise NotImplementedError

def svd_power_method_tall(A: callable,
                          AHA: callable,
                          rank: int,
                          inp_dims: tuple,
                          niter: Optional[int] = 100,
                          inp_dtype: Optional[torch.dtype] = torch.complex64,
                          device: Optional[torch.device] = torch.device('cpu'),
                          verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD on operator A using power method. This returns the compact SVD.
    http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf

    Parameters:
    -----------
    A : callable
        linear operator maps from M to N, assuming M > N
    AHA : callable
        linear operator maps from N to N, AHA = hermetian(A) * A
    rank : int
        rank of the SVD
    inp_dims : tuple
        input dimensions of A, corresponds to N but can be tuple
    inp_dtype : torch.dtype
        input data type
    device : torch.device
        device to run on
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    U : torch.Tensor
        left singular vectors with shape (out_dims, rank)
    S : torch.Tensor
        singular values with shape (rank)
    Vh : torch.Tensor
        right singular vectors with shape (rank, inp_dims)
    """

    # Consts
    # delta = 0.001
    # epsilon = 0.97
    # lamda = .1
    # N = torch.prod(torch.tensor(inp_dims)).item()
    # niter = round((torch.log(
    #     4 * torch.log(torch.tensor(2 * N / delta)) / (epsilon * delta)) / (2 * lamda)).item())

    # SVD return params
    U = []
    S = []
    V = []

    # Residual Operators
    def A_resid_operator(x):
        y = A(x)
        for i in range(len(U)):
            rank1_op = U[i] * torch.sum(V[i].conj() * x * S[i])
            y = y - rank1_op
        return y
    def AHA_resid_operator(x):
        y = AHA(x)
        for i in range(len(U)):
            rank1_op = V[i] * torch.sum(V[i].conj() * x * (S[i] ** 2))
            y = y - rank1_op
        return y
    
    for r in tqdm(range(rank), 'SVD Iterations', disable=not verbose):
        # Power method to calc u_i sigma_i v_i
        x0 = torch.randn(inp_dims, device=device, dtype=inp_dtype)
        v, _ = power_method_operator(AHA_resid_operator, x0, num_iter=niter, verbose=False)
        v = v / torch.linalg.norm(v)
        u = A_resid_operator(v)
        sigma = torch.linalg.norm(u)
        u = u / sigma

        # Append to list
        U.append(u)
        S.append(sigma)
        V.append(v)

    U = torch.stack(U, dim=-1)
    S = torch.tensor(S, device=device)
    Vh = torch.stack(V, dim=0).conj()

    return U, S, Vh

def power_method_matrix(M: torch.Tensor,
                        vec_init: Optional[torch.Tensor] = None,
                        num_iter: Optional[int] = 100,
                        verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses power method to find largest eigenvalue and corresponding eigenvector

    Parameters:
    -----------
    M : torch.Tensor
        input matrix with shape (..., n, n)
    vec_init : torch.Tensor
        initial guess of eigenvector with shape (..., n)
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    eigen_vec : torch.Tensor
        eigenvector with shape (..., n)
    eigen_val : torch.Tensor
        eigenvalue with shape (...)
    """
    # Consts
    n = M.shape[-1]
    assert M.shape[-2] == n
    assert M.ndim >= 2

    if vec_init:
        eigen_vec = vec_init[..., None]
    else:
        eigen_vec = torch.ones((*M.shape[:-1], 1), device=M.device, dtype=M.dtype)
    for i in tqdm(range(num_iter), 'Power Iterations', disable=not verbose):
        eigen_vec = M @ eigen_vec
        eigen_val = torch.linalg.norm(eigen_vec, axis=-2, keepdims=True)
        eigen_vec = eigen_vec / eigen_val
    eigen_vec = rearrange(eigen_vec, '... n 1 -> n ...')
    
    return eigen_vec, eigen_val.squeeze()

def power_method_operator(A: callable,
                          x0: torch.Tensor,
                          num_iter: Optional[int] = 15,
                          verbose: Optional[bool] = True) -> Tuple[torch.Tensor, float]:
    """
    Uses power method to find largest eigenvalue and corresponding eigenvector

    Parameters:
    -----------
    A : callable
        linear operator
    vec_init : torch.Tensor
        initial guess of eigenvector with shape (*vec_shape)
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    eigen_vec : torch.Tensor
        eigenvector with shape (*vec_shape)
    eigen_val : float
        eigenvalue
    """
    
    for _ in tqdm(range(num_iter), 'Max Eigenvalue', disable=not verbose):
        
        z = A(x0)
        ll = torch.norm(z)
        x0 = z / ll
    
    if verbose:
        print(f'Max Eigenvalue = {ll}')
    
    return x0, ll.item()

def inverse_power_method_operator(A: callable,
                                  x0: torch.Tensor,
                                  num_iter: Optional[int] = 15,
                                  n_cg_iter: Optional[int] = 10,
                                  verbose: Optional[bool] = True) -> Tuple[torch.Tensor, float]:
    """
    Uses power method to find largest eigenvalue and corresponding eigenvector
    of A^-1

    Parameters:
    -----------
    A : callable
        linear operator
    x0 : torch.Tensor
        initial guess of eigenvector with shape (*vec_shape)
    num_iter : int
        number of iterations to run power method
    n_cg_iter : int
        number of iterations to run conjugate gradient
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    eigen_vec : torch.Tensor
        eigenvector with shape (*vec_shape)
    eigen_val : float
        eigenvalue
    """
    
    for _ in tqdm(range(num_iter), 'Max Eigenvalue', disable=not verbose):
        
        z = conjugate_gradient(A, x0, num_iters=n_cg_iter, verbose=False, lamda_l2=1e-5)
        ll = torch.linalg.norm(z)
        x0 = z / ll
    
    if verbose:
        print(f'Max Eigenvalue = {ll}')
    
    return x0, ll.item()

def lin_solve(AHA: torch.Tensor, 
              AHb: torch.Tensor, 
              lamda: Optional[float] = 0.0, 
              solver: Optional[int] = 'lstsq') -> torch.Tensor:
    """
    Solves (AHA + lamda I) @ x = AHb for x

    Parameters:
    -----------
    AHA : torch.Tensor
        square matrix with shape (..., n, n)
    AHb : torch.Tensor
        matrix with shape (..., n, m)
    lamda : float
        optional L2 regularization 
    solver : str
        'pinv' - pseudo inverse 
        'lstsq' - least squares
        'inv' - regular inverse
    
    Returns:
    --------
    x : torch.Tensor
        solution with shape (..., n, m)
    """
    I = torch.eye(AHA.shape[-1], dtype=AHA.dtype, device=AHA.device)
    tup = (AHA.ndim - 2) * (None,) + (slice(None),) * 2
    AHA += lamda * I[tup]
    if solver == 'lstsq_torch':
        x = torch.linalg.lstsq(AHA, AHb, rcond=None).solution
    elif solver == 'lstsq':
        n, m = AHb.shape[-2:]
        AHA_cp = torch_to_np(AHA).reshape(-1, n, n)
        AHb_cp = torch_to_np(AHb).reshape(-1, n, m)
        dev = sp.get_device(AHA_cp)
        with dev:
            x = dev.xp.zeros_like(AHb_cp)
            for i in range(AHA_cp.shape[0]):
                x[i] = dev.xp.linalg.lstsq(AHA_cp[i], AHb_cp[i], rcond=None)[0]
        x = np_to_torch(x).reshape(AHb.shape)
    elif solver == 'pinv':
        x = torch.linalg.pinv(AHA, hermitian=True) @ AHb
    elif solver == 'inv':
        x = torch.linalg.inv(AHA) @ AHb
    else:
        raise NotImplementedError
    return x

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

def gradient_descent(AHA: nn.Module,
                     AHb: torch.Tensor,
                     lr: float,
                     num_iters: Optional[int] = 100,
                     lamda_l2: Optional[float] = 0.0,
                     tolerance: Optional[float] = 1e-8,
                     return_resids: Optional[bool] = False,
                     verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Parameters:
    -----------
    AHA : nn.Module 
        Linear operator representing the gram/normal operator of A
    AHb : torch.tensor
        The A hermitian transpose times b
    lr : float
        learning rate
    num_iters : int 
        Max number of iterations.
    lamda_l2 : float
        Replaces AHA with AHA + lamda_l2 * I
    tolerance : float 
        Used as stopping criteria
    return_resids : bool
        toggles return of residuals
    verbose : bool
        toggles print statements
    
    Returns:
    ---------
    x : torch.tensor <complex>
        least squares estimate of x
    """
    x = AHb
    AHA_wrapper = lambda x : AHA(x) + lamda_l2 * x
    resids = []
    for i in tqdm(range(num_iters), 'GD Iterations', disable=not verbose):
        grad = AHA_wrapper(x) - AHb
        x = x - lr * grad
        nrm = torch.norm(grad)
        resids.append(nrm.item())
        if nrm < tolerance:
            break
    if return_resids:
        return resids, x
    else:
        return x

def conjugate_gradient(AHA: nn.Module, 
                       AHb: torch.Tensor, 
                       P: Optional[nn.Module] = None,
                       num_iters: Optional[int] = 10, 
                       lamda_l2: Optional[float] = 0.0,
                       tolerance: Optional[float] = 1e-8,
                       return_resids: Optional[bool] = False,
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
    return_resids : bool
        toggles return of residuals
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
    if num_iters == 0:
        return x0

    # Define iterative vars
    r = AHb - AHA_wrapper(x0)
    z = P(r)
    p = z.clone()

    resids = []

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
        rnrm = torch.norm(r)
        resids.append(rnrm.item())
        if rnrm < tolerance:
            break

        # Update z
        z = P(r)

        # Update p
        beta = torch.real(torch.sum(r.conj() * z)) / rz
        p = z + beta * p
    
    if return_resids:
        return resids, x0
    else:
        return x0