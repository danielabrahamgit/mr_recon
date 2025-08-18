import torch
import sigpy as sp
import torch.nn as nn

from tqdm import tqdm
from typing import Optional, Tuple
from einops import rearrange, einsum
from mr_recon.utils import torch_to_np, np_to_torch
from mr_recon.dtypes import complex_dtype
from sigpy.mri import pipe_menon_dcf

def soft_thresh(x: torch.Tensor,
                lamda: float) -> torch.Tensor:
    """
    Soft thresholding operator for complex numbers.
    """
    x = torch.sgn(x) * torch.maximum(x.abs() - lamda, x.abs() * 0)
    return x

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

def svd_operator(A: callable,
                 AHA: callable,
                 inp_example: torch.Tensor,
                 rank: int,
                 num_iter: Optional[int] = 15,
                 tol: Optional[float] = 1e-9,
                 lobpcg: Optional[bool] = True,
                 verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses power method or lobpcg to to eigen-step in SVD on a matrix operator.

    Parameters:
    -----------
    A : callable
        linear operator mapping from (N, *inp_shape) to (N, *out_shape)
        where N is a batch dimension
    AHA : callable
        linear operator square shape mapping from (N, *inp_shape) to (N, *inp_shape)
    inp_example : torch.Tensor
        example input tensor with shape (*inp_shape) (also contains device and dtype info)
    rank : int
        number of ordered svd terms
    num_iter : int
        number of iterations to run power method
        OR
        number of iterations to run lobpcg if lobpcg is True
    tol : float
        tolerance for convergence of lobpcg
    lobpcg : bool
        toggles use of lobpcg instead of power method
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    U : torch.Tensor
        left vectors with shape (*out_shape, rank)
    S : torch.Tensor
        singular values with shape (rank,)
    Vh : torch.Tensor
        right vectors with shape (rank, *inp_shape)
    """
    
    # Compute right singular vectors via eigen decomposition of AHA
    V, S = eigen_decomp_operator(AHA, inp_example, num_eigen=rank, num_iter=num_iter, tol=tol, lobpcg=lobpcg, verbose=verbose)
    
    # Sort by singular values
    idx = torch.argsort(S, descending=True)
    V = V[idx]
    S = S[idx] ** 0.5
    U = None
    
    # Clear GPU mem
    import gc
    gc.collect()
    with torch.cuda.device(inp_example.device):
        torch.cuda.empty_cache()
    
    # Compute left singular vectors
    bs = rank
    for l1 in tqdm(range(0, rank, bs), 'Calculating Left Singular Vectors', disable=not verbose):
        l2 = min(l1 + bs, rank)
        u = A(V[l1:l2])
        u = u.moveaxis(0, -1) / S[l1:l2]
        if U is None:
            U = u
        else:
            U = torch.cat((U, u), dim=-1)
    
    return U, S, V.conj()

def svd_power_method_tall(A: callable,
                          AHA: callable,
                          rank: int,
                          inp_dims: tuple,
                          niter: Optional[int] = 100,
                          inp_dtype: Optional[torch.dtype] = complex_dtype,
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

def svd_matrix_method_tall(A: torch.Tensor,
                           rank: int,
                           niter: Optional[int] = None,
                           verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD on matrix A using eigh matrix method. This returns the compact SVD.

    Parameters:
    -----------
    A : torch.Tensor
        input matrix with shape (N, M)
    rank : int
        rank of the SVD
    niter : int
        number of iterations to run lobpcg
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    U : torch.Tensor
        left singular vectors with shape (N, rank)
    S : torch.Tensor
        singular values with shape (rank)
    Vh : torch.Tensor
        right singular vectors with shape (rank, M)
    """
    # HACK -- torch lobpcg doesn't support complex numbers
    AHA = A.H @ A
    Ar_top = torch.cat([AHA.real, -AHA.imag], dim=1)
    Ar_bot = torch.cat([AHA.imag, AHA.real], dim=1)
    Ar = torch.cat([Ar_top, Ar_bot], dim=0)
    
    # Compute real eigen decomposition
    M = AHA.shape[-1]
    sig_squaredr, Vhr = torch.lobpcg(Ar, k=2*rank, niter=niter)
    
    # Sort by singular values
    idx = torch.argsort(sig_squaredr, descending=True)
    sig_squaredr = sig_squaredr[idx]
    Vhr = Vhr[..., idx]
    
    # Convert to complex
    Vh1 =       Vhr[..., :M, ::2] + 1j * Vhr[..., M:, ::2]
    Vh2 = -1j * Vhr[..., :M, 1::2] +     Vhr[..., M:, 1::2]
    sgns = torch.logical_and((Vh1.real * Vh2.real).mean(dim=-2) > 0, (Vh1.imag * Vh2.imag).mean(dim=-2) > 0)
    sgns = sgns.float() * 2 - 1
    Vh = (Vh1 + Vh2 * sgns)/2
    sig_squared = sig_squaredr[..., ::2]
    
    # Compute U
    Vh = Vh.H
    U = (A @ Vh.H / sig_squared ** 0.5)
    return U, sig_squared ** 0.5, Vh

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
                          num_iter: Optional[int] = 15,
                          tol: Optional[float] = 1e-9,
                          lobpcg: Optional[bool] = True,
                          largest: Optional[bool] = True,
                          verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses power method to find largest num_eigen eigenvalues and corresponding eigenvectors

    Parameters:
    -----------
    A : callable
        linear operator mapping from (N, *vec_shape) to (N, *vec_shape)
        where N is a batch dimension
    x0 : torch.Tensor
        initial guess of eigenvector with shape (*vec_shape)
    num_eigen : int
        number of eigenvalues to find
    num_iter : int
        number of iterations to run power method
        OR
        number of iterations to run lobpcg if lobpcg is True
    tol : float
        tolerance for convergence of lobpcg
    lobpcg : bool
        toggles use of lobpcg instead of power method
    largest : bool
        If True, computes the largest eigenpairs; otherwise, the smallest.
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    eigen_vecs : torch.Tensor
        eigenvectors with shape (num_eigen, *vec_shape)
    eigen_vals : torch.Tensor
        eigenvalues with shape (num_eigen,)
    """
    if lobpcg:
        
        # linops
        def matvec(x_flt):
            # x_flt (n, k) 
            x_vec = x_flt.T.reshape((x_flt.shape[1], *x0.shape))
            out_vec = A(x_vec) # k, *vec_shape
            out_flt = out_vec.reshape((x_flt.shape[1], -1)).T # (n, k)
            return out_flt
        
        # Run lobpcg
        n = x0.numel()
        k = num_eigen
        X = torch.randn((n, k), device=x0.device, dtype=x0.dtype)
        eigen_vals, eigen_vecs = lobpcg_operator(matvec, X, maxiter=num_iter, tol=tol, largest=largest, verbose=verbose)
        eigen_vecs = eigen_vecs.T.reshape((k, *x0.shape))
    else:
        eigen_vecs = torch.zeros(num_eigen, *x0.shape, device=x0.device, dtype=x0.dtype)
        eigen_vals = torch.zeros(num_eigen, device=x0.device, dtype=x0.dtype)
        A_clone = A
        
        def A_resid_operator(x):
            y = A_clone(x[None,])[0]
            VH_x = einsum(eigen_vecs.conj(), x, 'n ..., ... -> n') * eigen_vals
            V_diag_VH_x = einsum(eigen_vecs, VH_x, 'n ..., n -> ...')
            y = y - V_diag_VH_x
            return y
        
        for r in tqdm(range(num_eigen), 'Eigen Iterations', disable=not verbose):
            init = torch.randn_like(x0)
            init /= torch.linalg.norm(init)
            
            vec, val = power_method_operator(A_resid_operator, init, 
                                            num_iter=num_iter, 
                                            verbose=False)
            eigen_vecs[r] = vec
            eigen_vals[r] = val

    return eigen_vecs, eigen_vals

def lobpcg_operator(A: callable, 
                    X: torch.Tensor, 
                    precond: Optional[callable] = None, 
                    maxiter: Optional[int] = 100, 
                    tol: Optional[float] = 1e-6, 
                    largest: Optional[bool] = True,
                    verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Matrix-free LOBPCG for a symmetric operator using only matvec calls.
    
    Thank you chatGPT

    Parameters:
    -----------
    A : callable
        (n --> n) function that takes a tensor with shape (n, k) and return A(X) with shape (n, k).
    X : torch.Tensor
        Initial guess for eigenvectors, shape (n, k). (rows need not be orthonormal.)
    precond : callable or None
        Function that applies a preconditioner to a tensor.
        If None, no preconditioning is applied.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence based on the residual norm.
    largest : bool
        If True, computes the largest eigenpairs; otherwise, the smallest.
    verbose : bool
        If True, shows progress bar.

    Returns:
    --------
    evals : torch.Tensor
        Approximated eigenvalues with shape (k,)
    evecs : torch.Tensor
        Approximated eigenvectors with shape (n, k)
    """
    
    # Ensure X has orthonormal columns
    X, _ = torch.linalg.qr(X)
    n, k = X.shape

    # Compute initial A*X and perform a Rayleigh–Ritz on the subspace spanned by X.
    AX = A(X)
    T = X.H @ AX
    evals, eigvecs = torch.linalg.eigh(T)
    
    # torch.linalg.eigh returns eigenvalues in ascending order.
    if largest:
        idx = torch.argsort(evals, descending=True)
    else:
        idx = torch.argsort(evals)
    evals = evals[idx]
    eigvecs = eigvecs[:, idx]
    X = X @ eigvecs  # new approximations for eigenvectors
    AX = AX @ eigvecs

    # Optionally store a previous search direction (for subspace expansion)
    P = None

    for it in tqdm(range(maxiter), 'LOBPCG Iteration', disable=not verbose):
        # Compute the residual: R = A*X - X*Lambda
        R = AX - X * evals.unsqueeze(0)
        res_norm = torch.linalg.norm(R, dim=0)
        
        # Check convergence for each eigenpair.
        if torch.all(res_norm < tol):
            break

        # Apply preconditioning if available.
        if precond is not None:
            W = precond(R)
        else:
            W = R

        # Build an expanded search subspace.
        if P is not None:
            S = torch.cat([X, W, P], dim=1)
        else:
            S = torch.cat([X, W], dim=1)

        # Orthonormalize the subspace S.
        Q, _ = torch.linalg.qr(S)

        # Compute A*Q using the matvec operator.
        AQ = A(Q)
        T_sub = Q.H @ AQ
        
        # Solve the small eigenproblem.
        evals_sub, eigvecs_sub = torch.linalg.eigh(T_sub)
        if largest:
            idx = torch.argsort(evals_sub, descending=True)
        else:
            idx = torch.argsort(evals_sub)
        evals_sub = evals_sub[idx]
        eigvecs_sub = eigvecs_sub[:, idx]

        # Update our approximations: choose the first k eigenpairs.
        X_new = Q @ eigvecs_sub[:, :k]
        AX_new = AQ @ eigvecs_sub[:, :k]
        new_evals = evals_sub[:k]

        # Optionally update the search direction with the orthogonal complement of X.
        P = X_new - X @ (X.H @ X_new)
        if P.shape[1] > 0:
            P, _ = torch.linalg.qr(P)

        # Update variables for the next iteration.
        X = X_new
        AX = AX_new
        evals = new_evals
        

    return evals, X
          
def lin_solve(AHA: torch.Tensor, 
              AHb: torch.Tensor, 
              lamda: Optional[float] = 0.0, 
              solver: Optional[int] = 'solve') -> torch.Tensor:
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
    solver = solver.lower()
    if lamda > 0:
        I = torch.eye(AHA.shape[-1], dtype=AHA.dtype, device=AHA.device)
        tup = (AHA.ndim - 2) * (None,) + (slice(None),) * 2
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
          return_ptols: Optional[bool] = False,
          return_xs: Optional[bool] = False,
          x0: Optional[torch.Tensor] = None,
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
    return_ptols : bool
        toggles return of ptols
    verbose : bool
        toggles print statements

    Returns
    ---------
    x : torch.tensor
        Reconstructed tensor
    """

    # Start at AHb
    if x0 is None:
        x = AHb.clone()
    else:
        x = x0
    z = x.clone()
    
    if num_iters <= 0:
        return x
    
    ptols = []
    if return_xs:
        xs = []
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
        ptols.append(ptol.item())
        if ptol < ptol_exit:
            if verbose:
                print(f'Tolerance reached after {k+1} iterations, exiting FISTA')
            break
        if return_xs:
            xs.append(x.clone())
    
    if return_ptols:
        return ptols, x
    if return_xs:
        return xs, x
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
                       weights: Optional[torch.Tensor] = None,
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
    if weights:
        AHA_wrapper = lambda x : AHA(x) + lamda_l2 * weights * x
    else:
        AHA_wrapper = lambda x : AHA(x) + lamda_l2 * x

    # Start at AHb
    x0 = AHb.clone()
    if num_iters == 0:
        return x0

    # Define iterative vars
    r = AHb - AHA_wrapper(x0)
    z = P(r)
    p = z.clone()

    if return_resids:
        resids = []
        xs = [x0.clone()]

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
        if return_resids:
            resids.append(rnrm.item())
            xs.append(x0.clone())
        if rnrm < tolerance:
            break

        # Update z
        z = P(r)

        # Update p
        beta = torch.real(torch.sum(r.conj() * z)) / rz
        p = z + beta * p
    
    if return_resids:
        return resids, xs
    else:
        return x0