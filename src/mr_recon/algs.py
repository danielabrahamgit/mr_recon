from turtle import forward
import torch
import sigpy as sp
import torch.nn as nn

from tqdm import tqdm
from typing import Optional, Tuple
from einops import rearrange, einsum
from mr_recon.utils import torch_to_np, np_to_torch
from sigpy.mri import pipe_menon_dcf
# from mr_recon.fourier import sigpy_nufft

def density_compensation(trj: torch.Tensor,
                         im_size: tuple,
                         num_iters: Optional[int] = 30,
                         method='sigpy'):
    """
    Computes density compensation factor using Pipe Menon method.
    Copied from sigpy.

    Parameters:
    -----------
    trj : torch.Tensor
        k-space trajectory with shape (..., d), scaled from -Ni/2 to Ni/2
        where Ni = im_size[i]
    im_size : tuple
        image size 
    num_iters : int
        number of iterations
    method : str
        'sigpy' - uses sigpy implementation
        'CG_ksp' - Congugate gradient in k-space
        'CG_img' - Congugate gradient in image-space
    
    Returns:
    --------
    dcf : torch.Tensor
        density compensation factor with shape (...)
    """

    # Consts
    os = 1
    im_size_os = [i * os for i in im_size]
    method = method.lower()
    torch_dev = trj.device
    if 'cpu' in str(torch_dev):
        idx = -1
    else:
        idx = torch_dev.index

    # Sigpy dcf
    if method == 'sigpy':
        dcf = pipe_menon_dcf(torch_to_np(trj), im_size, 
                            device=sp.Device(idx), 
                            max_iter=num_iters)    
        dcf = np_to_torch(dcf)
    
    # # CG in k-space
    # elif method == 'cg_ksp':
    #     nft = sigpy_nufft(im_size_os, trj.device.index, oversamp=1.0, width=4, beta=8, apodize=False)
    #     def GHG(x):
    #         adj = nft.adjoint(x[None,], trj[None,])
    #         fwd = nft.forward(adj, trj[None,])[0].real
    #         return fwd
        
    #     # Fix scaling factor and run CG
    #     ones = torch.ones(trj.shape[:-1], device=torch_dev, dtype=torch.float32)
    #     delta = ones.flatten() * 0
    #     delta[0] = 1
    #     scale = GHG(delta.reshape(ones.shape)).abs().max()
    #     AHA = lambda x : GHG(x) / scale
    #     dcf = conjugate_gradient(AHA, ones, num_iters=num_iters, lamda_l2=1e0 * 0, verbose=True).abs()

    # elif method == 'cg_img':
    #     delta = torch.zeros(im_size_os, device=torch_dev, dtype=torch.complex64)
    #     slc = tuple([im_size_os[i] // 2 for i in range(len(im_size_os))])
    #     delta[slc] = 1

    #     nft = sigpy_nufft(im_size_os, trj.device.index, oversamp=os, width=4, beta=8, apodize=False)
    #     kerns = nft.calc_teoplitz_kernels(trj[None,], os_factor=2.0)
    #     # def FHF(x):
    #     #     fwd = nft.forward(x[None,], trj[None,])
    #     #     adj = nft.adjoint(fwd, trj[None,])[0]
    #     #     return adj
    #     def FHF(x):
    #         adj = nft.normal_toeplitz(x[None,None], kerns)[0,0]
    #         return adj
    #     def FHF_inv(x):
    #         adj = nft.normal_toeplitz(x[None,None], 1 / (kerns + 1e1))[0,0]
    #         return adj
    #     scale = FHF(delta).abs().max()
    #     AHA = lambda x : FHF(x) / scale
    #     print(scale)
    #     # dcf_img = conjugate_gradient(AHA, delta, num_iters=num_iters*0 + 20, lamda_l2=1e1, verbose=True)
    #     dcf_img = FHF_inv(delta)
    #     dcf = nft.forward(dcf_img[None,], trj[None,])[0].abs()
        
    dcf /= dcf.max()
    return dcf

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
    niter : int
        number of iterations to run power method
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

def eigen_decomp_operator(A: callable,
                          x0: torch.Tensor,
                          num_eigen: int,
                          num_power_iter: Optional[int] = 15,
                          reverse_order: Optional[bool] = False,
                          verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses power method to find largest num_erigen eigenvalues and corresponding eigenvectors

    Parameters:
    -----------
    A : callable
        linear operator square shape
    x0 : torch.Tensor
        initial guess of eigenvector with shape (*vec_shape)
    num_eigen : int
        number of eigenvalues to find
    num_power_iter : int
        number of iterations to run power method
    reverse_order : bool
        toggles reverse order of eigenvalues, uses inverse power method instead.
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    eigen_vecs : torch.Tensor
        eigenvectors with shape (num_eigen, *vec_shape)
    eigen_vals : torch.Tensor
        eigenvalues with shape (num_eigen,)
    """
    eigen_vecs = torch.zeros(num_eigen, *x0.shape, device=x0.device, dtype=x0.dtype)
    eigen_vals = torch.zeros(num_eigen, device=x0.device, dtype=x0.dtype)
    def A_resid_operator(x):
        y = A_clone(x)
        VH_x = einsum(eigen_vecs.conj(), x, 'n ..., ... -> n') * eigen_vals
        V_diag_VH_x = einsum(eigen_vecs, VH_x, 'n ..., n -> ...')
        y = y - V_diag_VH_x
        return y
    
    for r in tqdm(range(num_eigen), 'Eigen Iterations', disable=not verbose):
        init = torch.randn_like(x0)
        init /= torch.linalg.norm(init)
        if reverse_order:
            # vec, val = inverse_power_method_operator(A_resid_operator, init, 
            #                                         num_iter=num_power_iter, 
            #                                         verbose=False)
            # Get largest eval first
            if r == 0:
                _, val_max = power_method_operator(A, init, 
                                                num_iter=num_power_iter * 10, 
                                                verbose=False)
                val_max *= 1.0
                A_clone = lambda x : val_max * x - A(x)
            vec, val = power_method_operator(A_resid_operator, init, 
                                            num_iter=num_power_iter, 
                                            verbose=False)
            
        else:
            vec, val = power_method_operator(A_resid_operator, init, 
                                            num_iter=num_power_iter, 
                                            verbose=False)
        eigen_vecs[r] = vec
        eigen_vals[r] = val


    if reverse_order:
        eigen_vals = val_max - eigen_vals
        # eigen_vals = 1 / eigen_vals

    return eigen_vecs, eigen_vals
                          
def inverse_power_method_operator(A: callable,
                                  x0: torch.Tensor,
                                  num_iter: Optional[int] = 15,
                                  n_cg_iter: Optional[int] = 12,
                                  lamda_l2: Optional[float] = 0.0,
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
    
    # ray = lambda x : (x.conj() * A(x)).sum() / ((x.norm() ** 2))
    for _ in tqdm(range(num_iter), 'Max Eigenvalue', disable=not verbose):
        # mu = ray(x0)
        z = conjugate_gradient(A, x0, num_iters=n_cg_iter, verbose=False, lamda_l2=0)
        ll = z.norm()
        x0 = z / ll
    # ll = (x0.conj() * A(x0)).sum()
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
        'solve' - torch.linalg.solve
        'lstsq' - least squares
        'inv' - regular inverse
        'cg' - conjugate gradient
        'gd' - gradient descent
    
    Returns:
    --------
    x : torch.Tensor
        solution with shape (..., n, m)
    """
    I = torch.eye(AHA.shape[-1], dtype=AHA.dtype, device=AHA.device)
    tup = (AHA.ndim - 2) * (None,) + (slice(None),) * 2
    solver = solver.lower()
    if lamda > 0:
        AHA += lamda * I[tup]
    if solver == 'lstsq_torch':
        x = torch.linalg.lstsq(AHA, AHb).solution
    elif solver == 'solve':
        x = torch.linalg.solve(AHA, AHb)
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
    elif solver == 'cg':
        x = conjugate_gradient(lambda x : AHA @ x, AHb, num_iters=100, lamda_l2=lamda)
    elif solver == 'gd':
        x = gradient_descent(lambda x : AHA @ x, AHb, lr=1e-4*2, num_iters=10000, lamda_l2=lamda)
    else:
        raise NotImplementedError
    return x

def FISTA(AHA: nn.Module, 
          AHb: torch.Tensor, 
          proxg: callable, 
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

    for k in tqdm(range(0, num_iters), 'FISTA Iterations', disable=not verbose):

        x_old = x.clone()
        x     = z.clone()
        
        gr    = AHA(x) - AHb
        x     = proxg(x - gr)
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