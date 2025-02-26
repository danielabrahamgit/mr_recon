import torch

from typing import Optional
from einops import einsum
from mr_recon import dtypes

def b0_est_naive(multi_echo_imgs: torch.Tensor,
                 echo_time_diff: float,
                 phase_unwrap: Optional[str] = None,
                 mask_thresh: Optional[float] = 0.01) -> torch.Tensor:
    """
    Estimated b0 map from multi-echo image data using naive method.

    Parameters:
    -----------
    multi_echo_imgs: torch.Tensor
        Multi-echo image data with shape (nechos, *im_size)
    echo_time_diff: float
        time difference between consecutive echos
    phase_unwrap: str
        Phase unwrapping method from
        'spatial' - unwrapping in spatial domain
        'temporal' - unwrapping in temporal domain
        None - no unwrapping
    mask_thresh: float
        Thresholds low signal regions by taking mask_thresh * max(signal) from first echo to be zero

    Returns:
    --------
    b0_map: torch.Tensor
        Estimated b0 map with shape (*im_size)
    """

    # Constants
    nechos = multi_echo_imgs.shape[0]
    im_size = multi_echo_imgs.shape[1:]
    assert nechos > 1, 'At least two echos are required for B0 estimation.'

    # Magnitude mask 
    first_echo = multi_echo_imgs[0]
    mask = (first_echo.abs() / first_echo.abs().max()) > mask_thresh

    # Phase images relative to first echo
    eps = 1e-9
    phase_imgs = (multi_echo_imgs[1:] / (first_echo[None,] + eps)).angle()
    if phase_unwrap is not None:
        raise NotImplementedError
    
    # Fit linear phase to phase images
    b0_map = torch.zeros(im_size, dtype=dtypes.real_dtype, device=phase_imgs.device)
    n = torch.arange(1, nechos, dtype=dtypes.real_dtype, device=phase_imgs.device)
    b0_map = -einsum(phase_imgs, n, 'N ..., N -> ...') / (n @ n) / (2 * torch.pi * echo_time_diff)
    b0_map = b0_map * mask

    return b0_map

def b0_est_regularized(multi_echo_imgs: torch.Tensor,
                       echo_time_diff: float,
                       proxg: Optional[callable] = None,
                       step_size: Optional[float] = 1e-2,
                       niter: Optional[int] = 100,
                       method: Optional[str] = 'pgd') -> torch.Tensor:
    """
    Estimated b0 map from multi-echo image data using non-linear least squares 
    with spatial regularization

    multi_echo_imgs[i] = exp(-j2pi b0_map * TE_i) exp(- R2_map * TE_i) * img + noise

    Parameters:
    -----------
    multi_echo_imgs: torch.Tensor
        Multi-echo image data with shape (nechos, *im_size)
    echo_time_diff: float
        time difference between consecutive echos
    proxg: callable
        proximal operator for spatial regularization. Takes input and step size as arguments
    step_size: float
        Step size for iterative solve
    niter: int
        Number of iterations for iterative solve
    method: str
        Optimization method for non-linear least squares
        'gd' - Gradient descent -- proxg is actually just gradient of regularizer in this case
        'pgd' - Proximal gradient descent
        'apgd' - Accelerated proximal gradient descent

    Returns:
    --------
    b0_map: torch.Tensor
        Estimated b0 map with shape (*im_size)
    """ 

    # Constants
    nechos = multi_echo_imgs.shape[0]
    im_size = multi_echo_imgs.shape[1:]
    assert nechos > 1, 'At least two echos are required for B0 estimation.'
    deltas = torch.arange(nechos, dtype=dtypes.real_dtype, device=multi_echo_imgs.device) * echo_time_diff

    # Defaults
    if proxg is None:
        proxg = lambda inp, step : inp

    # maximum likelyhood loss func
    eps = 1e-9
    mag_sq = multi_echo_imgs.abs() ** 2
    all_prods = einsum(mag_sq, mag_sq, 'N1 ..., N2 ... -> N1 N2 ...')
    all_angs = torch.angle(einsum(multi_echo_imgs, 1 / (multi_echo_imgs + eps), 'N1 ..., N2 ... -> N1 N2 ...'))
    tup = 2 * (slice(None),) + len(im_size) * (None,)
    all_time_diffs = (deltas[:, None] - deltas[None, :])[tup]
    denom = torch.sum(mag_sq, dim=0) + 1e-6
    def ml_loss(b0_rad):
        cos_term = 1 - torch.cos(all_angs + b0_rad * all_time_diffs)
        num = torch.sum(all_prods * cos_term, dim=(0, 1))
        return torch.sum(num / denom)
    
    # gradient of ml_loss wrt b0_rad
    def ml_grad(b0_rad):
        sin_term = torch.sin(all_angs + b0_rad * all_time_diffs) * all_time_diffs
        num = torch.sum(all_prods * sin_term, dim=(0, 1))
        return num / denom
        
    # Initialize with naive estimate
    b0_hz = b0_est_naive(multi_echo_imgs, echo_time_diff) * 0
    b0_rad = b0_hz * 2 * torch.pi
    
    # Proximal gradient descent
    if method == 'gd':
        for i in range(niter):
            grad_step = b0_rad - step_size * ml_grad(b0_rad) - step_size * proxg(b0_rad, 1)
            b0_rad = grad_step
    elif method == 'pgd':
        for i in range(niter):
            grad_step = b0_rad - step_size * ml_grad(b0_rad)
            b0_rad = proxg(grad_step, step_size)
    # Accelerated proximal gradient descent
    elif method == 'apgd':
        t = 1
        p = b0_rad.clone()
        x = b0_rad
        x_prev = b0_rad.clone()
        for i in range(niter):
            t_next = (1 + (1 + 4 * t ** 2) ** 0.5) / 2
            beta = (t - 1) / t_next
            p = b0_rad + beta * (x - x_prev)
            x_prev = x.clone()
            x = proxg(p - step_size * ml_grad(p), step_size)
        b0_rad = x
    else:
        raise NotImplementedError
    
    b0_map = b0_rad / (2 * torch.pi)
    return b0_map