import torch

from tqdm import tqdm
from typing import Optional
from mr_recon.algs import conjugate_gradient, power_method_operator, inverse_power_method_operator
from mr_recon.utils import gen_grd

def calc_variance_mc(AHA: callable,
                     AH: callable,
                     ksp: torch.Tensor,
                     sigma_noise: Optional[float] = 1e-2,
                     n_monte_carlo: Optional[int] = 1000) -> torch.Tensor:
    """
    Computes variance map for a given forward gram operator and k-space data.

    Parameters
    ----------
    AHA : callable
        A callable that applies the gram operator to image
    AH : callable
        A callable that applies the adjoint of the forward operator to k-space data.
    ksp : torch.Tensor
        k-space data.
    sigma_noise : float
        Standard deviation of noise in k-space data.
    n_monte_carlo : int
        Number of monte carlo samples to use.
    
    Returns
    -------
    var : torch.Tensor
        Variance map.
    """

    recons = []
    for _ in tqdm(range(n_monte_carlo), 'G-factor Loop'):
        ksp_noisy = ksp + sigma_noise * (torch.randn_like(ksp) + 1j * torch.randn_like(ksp))
        recons.append(conjugate_gradient(AHA, AH(ksp_noisy), num_iters=15, verbose=False))
    var = torch.var(torch.stack(recons, dim=0), dim=0)

    return var

def calc_variance_analytic(AHA: callable,
                           im_size: tuple,
                           device: torch.device,
                           sigma_noise: Optional[float] = 1e-2,
                           msk: Optional[torch.Tensor] = None,
                           approx_batch_size: Optional[int] = 100) -> torch.Tensor:
    """
    Computes variance map using analytic method. 

    recon image = (AHA)^-1 AH(b + n)

    means that recon image ~ N((AHA)^-1 AH b, (AHA)^-1 sigma^2)

    So, we just need the diagonal term of (AHA)^-1 sigma^2

    Parameters
    ----------
    AHA : callable
        A callable that applies the gram operator to image
    im_size : tuple
        Size of the image.
    device : torch.device
        Device to use for computation.
    sigma_noise : float
        Standard deviation of noise in k-space data.
    msk : torch.Tensor
        Image domain mask
    approx_batch_size : int
        Batch size for aproximate computation.
    """

    all_inds = gen_grd(im_size, fovs=im_size).type(torch.int32).reshape(-1, 2).to(device)
    if msk is not None:
        all_inds = all_inds[msk.reshape(-1) > 0]
    all_inds[..., 0] += im_size[0] // 2
    all_inds[..., 1] += im_size[1] // 2
    all_inds = all_inds[torch.randperm(all_inds.shape[0], device=device)]
    delta = torch.zeros(im_size, device=device, dtype=torch.complex64)
    var = torch.ones(im_size, device=device, dtype=torch.float32)
    for i1 in tqdm(range(0, all_inds.shape[0], approx_batch_size), 'G-factor Loop'):
        i2 = min(i1 + approx_batch_size, all_inds.shape[0])
        rows = all_inds[i1:i2, 0]
        cols = all_inds[i1:i2, 1]
        delta *= 0.0
        delta[rows, cols] = 1.0
        x = conjugate_gradient(AHA, delta, num_iters=15, verbose=False)
        var[rows, cols] = x[rows, cols].abs()
    var = var * sigma_noise ** 2

    return var

def calc_variance_eigen(AHA: callable,
                        im_size: tuple,
                        device: torch.device,
                        sigma_noise: Optional[float] = 1e-2,
                        msk: Optional[torch.Tensor] = None,
                        n_eigen: Optional[int] = 20) -> torch.Tensor:
    """
    Computes variance map using analytic eigen method. 

    recon image = (AHA)^-1 AH(b + n)

    means that recon image ~ N((AHA)^-1 AH b, (AHA)^-1 sigma^2)

    So, we just need the diagonal term of (AHA)^-1 sigma^2.
    We will find the n_eigen smallest eigen vectors of AHA, 
    and use them to approximate the variance.

    Parameters
    ----------
    AHA : callable
        A callable that applies the gram operator to image
    im_size : tuple
        Size of the image.
    device : torch.device
        Device to use for computation.
    sigma_noise : float
        Standard deviation of noise in k-space data.
    msk : torch.Tensor
        Image domain mask
    n_eigen : int
        number of smallest eigenvalues to copmute
    """

    # Inverse eigen decomp
    vecs = []
    vals = []
    def AHA_resid_operator(x):
        y = AHA(x)
        for i in range(len(vecs)):
            rank1_op = vecs[i] * torch.sum(vecs[i].conj() * x * vals[i])
            y = y - rank1_op
        return y
    for i in range(n_eigen):
        x0 = torch.randn(im_size, device=device, dtype=torch.complex64)
        v, lamda = inverse_power_method_operator(AHA_resid_operator, x0, num_iter=15, n_cg_iter=15)
        vecs.append(v)
        vals.append(lamda)
        
    return 