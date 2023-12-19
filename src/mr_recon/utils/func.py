import torch
import cupy as cp
import sigpy as sp
import numpy as np
import torch.fft as fft

from typing import Optional
from scipy.signal import get_window

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

def apply_window(ksp_rect: np.ndarray, 
                 window_func: Optional[str] = 'hamming') -> np.ndarray:
    """
    Applies windowing function to a rect-linear k-space signal

    Parameters:
    -----------
    ksp_rect : np.ndarray <complex64>
        k-space rect region with shape (B, N_{ndim-1}, ..., N_1, N_0)
        B is batch dim
    window_func : str
        windowing function from scipy.signal.windows
    
    Returns:
    --------
    ksp_rect_win : np.ndarray <complex64>
        windowed k-space with same shape as ksp_cal
    
    """
    dev = sp.get_device(ksp_rect)
    with dev:
        win = sp.to_device(np.ones(ksp_rect.shape[1:]), dev)
        ksp_rect_win = ksp_rect.copy()
        for i in range(win.ndim):
            tup = i * (None,) + (slice(None),) + (None,) * (win.ndim - 1 - i)
            win *= sp.to_device(get_window(window_func, ksp_rect.shape[1]), dev)[tup]
        ksp_rect_win *= win[None,]
    return ksp_rect_win