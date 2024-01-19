import torch
import cupy as cp
import sigpy as sp
import numpy as np
import torch.fft as fft

from typing import Optional, Union
from scipy.signal import get_window
from einops import einsum

def fourier_resize(x, new_shape):

    # Get GPU dev
    x_np = torch_to_np(x)
    dev = sp.get_device(x_np)
    
    with dev:

        # FFT
        ndim = len(new_shape)
        X = sp.fft(x_np, axes=tuple(range(-ndim, 0)))

        # Zero pad/chop
        oshape = X.shape[:-ndim] + new_shape
        X_rs = sp.resize(X, oshape)

        # Windowing
        X_rs = apply_window(X_rs, ndim, 'hamming')

        # IFFT
        x_rs = sp.ifft(X_rs, axes=tuple(range(-ndim, 0)))

    # Convert to original
    if torch.is_tensor(x):
        return np_to_torch(x_rs)
    else:
        return x_rs

def calc_coil_subspace(ksp_mat: np.array,
                       new_coil_size: Union[int, float],
                       *args):

    # Estimate coil subspace
    n_coil = ksp_mat.shape[0]
    u, s, vt = np.linalg.svd(ksp_mat.reshape((n_coil, -1)), full_matrices=False)
    if new_coil_size is None:
        new_coil_size = n_coil
    elif type(new_coil_size) is float:
        cmsm = np.cumsum(s)
        n_coil = np.argwhere(cmsm > new_coil_size * cmsm[-1]).squeeze()[0]
    elif type(new_coil_size) is int:
        n_coil = new_coil_size
    coil_subspace = u[:, :n_coil].conj()

    comps = []
    for arg in args:
        comps.append(einsum(arg, coil_subspace, 'nc ..., nc nc2 -> nc2 ...'))

    if len(comps) == 0:
        return coil_subspace

    else:
        return [coil_subspace] + comps

def lin_solve(AHA: torch.Tensor, 
              AHb: torch.Tensor, 
              lamda: Optional[float] = 0.0, 
              solver: Optional[int] = 'lstsq'):
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
    if solver == 'lstsq':
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

def normalize(shifted, target, ofs=True, mag=False, return_params=False):
    if mag:
        col1 = np.abs(shifted).flatten()
        y = np.abs(target).flatten()
    else:
        col1 = shifted.flatten()
        y = target.flatten()

    if ofs:
        col2 = col1 * 0 + 1
        A = np.array([col1, col2]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        b = 0
        a = np.linalg.lstsq(np.array([col1]).T, y, rcond=None)[0]

    if return_params:
        return a * shifted + b, (a, b)
    else:
        return a * shifted + b

def np_to_torch(*args):
    ret_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret_args.append(torch.as_tensor(arg))
        elif isinstance(arg, cp.ndarray):
            ret_args.append(torch.as_tensor(arg, device=torch.device(int(arg.device))))
        elif isinstance(arg, torch.Tensor):
            ret_args.append(arg)
        else:
            ret_args.append(None)

    if len(ret_args) == 1:
        ret_args = ret_args[0]

    return ret_args

def torch_to_np(*args):
    ret_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.is_cuda:
                with cp.cuda.Device(arg.get_device()):
                    ret_args.append(cp.asarray(arg))
            else:
                ret_args.append(arg.numpy())
        elif isinstance(arg, np.ndarray) or isinstance(arg, cp.ndarray):
            ret_args.append(arg)
        else:
            ret_args.append(None)
    
    if len(ret_args) == 1:
        ret_args = ret_args[0]

    return ret_args

def batch_iterator(total, batch_size):
    assert total > 0, f'batch_iterator called with {total} elements'
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])

def sp_fft(x, dim=None, oshape=None):
    """Matches Sigpy's fft, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, s=oshape, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x

def sp_ifft(x, dim=None, oshape=None):
    """Matches Sigpy's fft adjoint, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.ifftn(x, s=oshape, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x

def apply_window(sig: np.ndarray, 
                 ndim: int,
                 window_func: Optional[str] = 'hamming') -> np.ndarray:
    """
    Applies windowing function to a rect-linear k-space signal

    Parameters:
    -----------
    sig : np.ndarray <complex64>
        signal with shape (..., N_{ndim-1}, ..., N_1, N_0)
    ndim : int
        apply windowing on last ndim dimensions
    window_func : str
        windowing function from scipy.signal.windows
    
    Returns:
    --------
    sig_win : np.ndarray <complex64>
        windowed signal with same shape as sig
    
    """
    dev = sp.get_device(sig)
    with dev:
        win = sp.to_device(np.ones(sig.shape[-ndim:]), dev)
        sig_win = sig.copy()
        dim_ofs = sig.ndim - ndim
        for i in range(win.ndim):
            tup = i * (None,) + (slice(None),) + (None,) * (win.ndim - 1 - i)
            win *= sp.to_device(get_window(window_func, sig.shape[dim_ofs + i]), dev)[tup]
        tup = (None,) * (dim_ofs) + (slice(None),) * ndim
        sig_win *= win[tup]
    return sig_win