import torch

from tqdm import tqdm
from einops import einsum
from typing import Optional
from mr_recon.algs import eigen_decomp_operator

def spatial_std_svd(AHA: callable,
                    im_size: tuple,
                    lambda_l2: Optional[float] = 0.0,
                    dtype: Optional[torch.dtype] = torch.complex64,
                    device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """
    Suppose you have a SENSE model
    
    min_x ||Ax - b||_2^2

    x_hat = (A^H A + lambda I)^-1 A^H b

    where A is the forward operator, A^H is the adjoint operator, and b is the data.

    The g-factor is simply cov((A^H A + lambda I)^-1 A^H (noise)) 
    which is U (S / (S + lambda I)^2) UH where AHA = U S UH

    Parameters
    ----------
    AHA : callable
        Function that computes AHA
    im_size : tuple
        Image size
    lambda_l2 : float, optional
        L2 regularization parameter
    dtype : torch.dtype, optional
        Data type of input image
    device : torch.device, optional
        Device of input image
    
    Returns
    -------
    spatial_std : torch.Tensor
        Spatial standard deviation with shape (*im_size)
    """

    # Compute SVD of AHA
    x0 = torch.randn(im_size, dtype=dtype, device=device)
    evecs, evals = eigen_decomp_operator(AHA, x0, 
                                         num_eigen=10,
                                         num_power_iter=25,
                                         reverse_order=True)
    # evals_new = 1 / evals
    evals_new = evals
    std_spatial = einsum(evecs * evecs.conj(), evals_new / ((evals_new + lambda_l2) ** 2), 'neig ..., neig -> ...')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(evals.abs().cpu())
    return std_spatial.abs()

def mean_std_monte_carlo(recon_alg: callable,
                         data: torch.Tensor,
                         im_size: tuple,
                         n_monte_carlo: Optional[int] = 100,
                         sigma: Optional[float] = 1e-3,
                         verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Compute the spatial mean and standard deviation using Monte Carlo simulations.

    Parameters
    ----------
    recon_alg : callable
        Reconstruction algorithm, takes in data, outputs image
    data : torch.Tensor
        Data to reconstruct
    im_size : tuple
        Image size
    n_monte_carlo : int, optional
        Number of Monte Carlo simulations
    sigma : float, optional
        Noise level
    verbose : bool, optional
        Print progress

    Returns
    -------
    mean : torch.Tensor
        Spatial mean with shape (*im_size)
    std : torch.Tensor
        Spatial standard deviation with shape (*im_size)
    """
    dtype = data.dtype
    device = data.device
    recons = torch.zeros(n_monte_carlo, *im_size, dtype=dtype, device=device)
    for n in tqdm(range(n_monte_carlo), 'Monte Carlo Iterations', disable=not verbose):
        data_noisy = data + torch.randn_like(data) * sigma
        recons[n] = recon_alg(data_noisy)
    return recons.mean(dim=0), recons.std(dim=0)