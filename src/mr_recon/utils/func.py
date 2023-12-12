import torch
import cupy as cp
import sigpy as sp
import numpy as np
import torch.fft as fft

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
