import torch
import numpy as np
import torch.nn.functional as F

from mr_recon.fourier import sigpy_nufft
from einops import rearrange, einsum
from typing import Optional
from tqdm import tqdm

def normalize_phis(phis: torch.Tensor) -> torch.Tensor:
    """
    Normalize the phis to be between -1/2 and 1/2

    Args:
    -----
    phis : torch.Tensor
        phase basis functions with shape (B, *im_size)
    

    Returns:
    --------
    torch.Tensor
        normalized phis with the same shape as input in range [-1/2, 1/2]
    """
    phi_scales = phis.abs().view(phis.shape[0], -1).max(dim=-1).values * 2 # B 
    tup = (slice(None),) + (None,) * (phis.ndim - 1)
    return phis / phi_scales[tup]

def gen_random_alphas(oshape: tuple,
                      scale: Optional[float] = 0.1,
                      torch_dev: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """
    Generate random alphas for the optimization process.

    Args:
    -----
    oshape : tuple
        shape of the output alphas
    scale : float
        scaling factor for the random alphas, default is 0.1
    torch_dev : torch.device
        device to create the tensor on, default is 'cpu'

    Returns:
    --------
    torch.Tensor
        random alphas with shape oshape
    """
    return torch.randn(oshape, device=torch_dev) * scale

def eddy_focus_metric(img_recon: torch.Tensor,
                      phase_guesses: torch.Tensor,
                      trj: torch.Tensor) -> torch.Tensor:
    """
    Compute the temporal metric for the eddy focus optimization.
    
    Args:
    -----
    img_recon : torch.Tensor
        reconstructed image with background phase removed, shape (*im_size)
    phase_guesses : torch.Tensor
        guessed phase with shape (K, *im_size)
    trj : torch.Tensor
        trajectory with shape (T, 2)
    
    Returns:
    --------
    torch.Tensor
        temporal metric with shape (K, *trj_size)
    """
    
    # Compute the k-space for the guessed phase at trj and -trj
    nufft = sigpy_nufft(img_recon.shape)
    kspace = nufft.forward((img_recon * phase_guesses.conj())[None,], trj[None,])[0]
    kspace_flip = nufft.forward((img_recon * phase_guesses.conj())[None,], -trj[None,])[0]
    
    # Compute the metric as the squared difference between kspace and its conjugate
    metric = ((kspace.conj() - kspace_flip).abs().square())
    
    return metric

def build_alpha_trajectory(img_recon: torch.Tensor,
                           phis: torch.Tensor,
                           trj: torch.Tensor,
                           window_size: Optional[int] = 100,
                           num_guesses: Optional[int] = 50,
                           noise_scale: Optional[float] = 0.01,) -> torch.Tensor:
    """
    Build the alpha trajectory for the optimization process.
    
    Args:
    -----
    img_recon : torch.Tensor
        reconstructed image with background phase removed, shape (*im_size)
    phis : torch.Tensor
        phase basis functions with shape (B, *im_size)
    trj : torch.Tensor
        trajectory with shape (T, 2)
    window_size : int
        size of the sliding window
    num_guesses : int
        number of random guesses for the alphas
    noise_scale : float
        scaling factor for the guessing random alphas
    
    Returns:
    --------
    torch.Tensor
        optimized alphas with shape (T, B)
    """
    
    # Consts
    torch_dev = img_recon.device
    im_size = phis.shape[1:]
    T, B = trj.shape[0], phis.shape[0]
    W = window_size
    L = num_guesses
    assert trj.shape[1] == 2
    assert trj.ndim == 2
    assert phis.ndim == 3
    
    # Initialize to zeros
    alphas_recon = torch.zeros((T, B), device=torch_dev, dtype=torch.float32)
    
    # Loop over time
    for t in tqdm(range(W, T), 'Estimating coefficients over trajectory'):
        
        # Window of size W
        t_slc = slice(t - W, t)
        
        # Mean of the previous alphas in window
        avg_alpha = alphas_recon[t_slc].mean(dim=0) # B,
        
        # Guess new ones at some random locations relative to the mean
        alpha_guesses = gen_random_alphas((L, B), scale=noise_scale, torch_dev=torch_dev) + avg_alpha
        phase_guesses = einsum(alpha_guesses, phis, 'L B, B ... -> L ...')
        phase_guesses = torch.exp(-2j * np.pi * phase_guesses) # L, *im_size
        
        # Find best guess over metrics
        metrics = eddy_focus_metric(img_recon, phase_guesses, trj[t_slc]).mean(dim=1)
        idx = torch.argmin(metrics, dim=0)
        alphas_recon[t] = alpha_guesses[idx]
    
    return alphas_recon

def design_lowpass_kernel(cutoff, fs, kernel_size=101, window='hamming'):
    """
    Design a 1D FIR lowpass filter kernel using a windowed sinc function.
    
    Parameters:
      cutoff : float
          The cutoff frequency in Hz.
      fs : float
          The sampling rate in Hz.
      kernel_size : int, optional
          Number of filter taps (must be odd for symmetry). Default is 101.
      window : str, optional
          Type of window to use. Currently only 'hamming' is supported.
          
    Returns:
      kernel : torch.Tensor
          A 1D tensor containing the filter kernel.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric FIR filter.")
    
    # Normalized cutoff frequency (as a fraction of the Nyquist frequency)
    # Note: For an ideal lowpass filter using sinc, we need the cutoff as a fraction of fs.
    # The ideal impulse response is: h[n] = 2*cutoff/fs * sinc(2*cutoff/fs*(n - M/2))
    W = 2 * cutoff / fs  # normalized cutoff frequency (0 < W < 1)
    
    # Create an index tensor centered around zero
    M = kernel_size
    n = torch.arange(0, M, dtype=torch.float32) - (M - 1) / 2.0

    # Compute the ideal sinc filter; note that torch.where handles the singularity at 0
    # sinc(x) in PyTorch is defined as sin(pi*x)/(pi*x), so adjust accordingly.
    h = W * torch.where(n == 0, torch.tensor(1.0), torch.sin(torch.pi * W * n) / (torch.pi * W * n))
    
    # Apply window: Hamming window
    if window.lower() == 'hamming':
        ham = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(0, M, dtype=torch.float32) / (M - 1))
        h = h * ham
    else:
        raise ValueError("Unsupported window type")
    
    # Normalize the kernel so that its sum equals 1.
    h = h / h.sum()
    return h

def lowpass_filter_torch(signal, cutoff, fs, dim=-1, kernel_size=101, window='hamming'):
    """
    Apply a lowpass FIR filter to a multi-dimensional tensor along one dimension.
    
    Parameters:
      signal : torch.Tensor
          The input tensor (can be multi-dimensional).
      cutoff : float
          The cutoff frequency in Hz.
      fs : float
          The sampling rate in Hz.
      dim : int, optional
          The dimension along which to apply the filter (default is last dimension).
      kernel_size : int, optional
          Number of filter taps (default is 101, must be odd).
      window : str, optional
          Type of window to use (default is 'hamming').
          
    Returns:
      filtered_signal : torch.Tensor
          The filtered tensor, with the same shape as input.
    """
    # Design the 1D filter kernel and reshape for conv1d: shape (out_channels, in_channels, kernel_size)
    kernel = design_lowpass_kernel(cutoff, fs, kernel_size, window).to(signal.device)
    # In conv1d we use groups to apply the same filter to multiple channels.
    # We need to add dimensions: [1, 1, kernel_size]
    kernel = kernel.view(1, 1, -1)

    # Move the filtering dimension to the last position if it isn't already
    if dim != -1 and dim != signal.dim() - 1:
        perm = list(range(signal.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        signal = signal.permute(perm)

    # Save original shape information
    original_shape = signal.shape
    # Merge all dimensions except the last one into batch dimension for conv1d.
    signal = signal.contiguous().view(-1, 1, original_shape[-1])

    # Apply 1D convolution. Use padding='same' behavior manually:
    pad = (kernel_size - 1) // 2
    filtered = F.conv1d(signal, kernel, padding=pad, groups=1)

    # Reshape back to original dimensions.
    filtered = filtered.view(*original_shape)
    
    # If we permuted earlier, reverse the permutation.
    if dim != -1 and dim != (len(original_shape) - 1):
        inv_perm = list(range(len(original_shape)))
        inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
        filtered = filtered.permute(inv_perm)
    
    return filtered
