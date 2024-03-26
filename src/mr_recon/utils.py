import torch
import cupy as cp
import sigpy as sp
import numpy as np

from typing import Optional
from fast_pytorch_kmeans import KMeans
from scipy.signal import get_window

def quantize_data(data: torch.Tensor,  
                  K: int,
                  method: Optional['str'] = 'uniform') -> torch.Tensor:
    """
    Given data of shape (..., d), finds K 'clusters' with shape (K, d)

    Parameters:
    -----------
    data : torch.Tensor
        data to quantize with shape (..., d)
    K : int
        number of clusters/quantization centers
    method : str
        selects the quantization method
        'cluster' - uses k-means to optimally find centers
        'uniform' - uniformly spaced bins
    
    Returns:
    --------
    centers : torch.Tensor
        cluster/quantization centers with shape (K, d)
    idxs : torch.Tensor <int>
        indices of the closest center for each point with shape (...)
    """

    # Consts
    torch_dev = data.device
    d = data.shape[-1]
    data_flt = data.reshape((-1, d))

    # Cluster
    if method == 'cluster':
        max_iter = 1000
        mode = 'euclidean'
        verbose = 1
        # mode = 'cosine'
        if (torch_dev.index == -1) or (torch_dev.index is None):
            kmeans = KMeans(n_clusters=K,
                            max_iter=max_iter,
                            verbose=verbose,
                            mode=mode)
            idxs = kmeans.fit_predict(data_flt)
        else:
            with torch.cuda.device(torch_dev):
                kmeans = KMeans(n_clusters=K,
                                max_iter=max_iter,
                                verbose=verbose,
                                mode=mode)
                idxs = kmeans.fit_predict(data_flt)
        centers = kmeans.centroids

    # Uniformly spaced time segments
    else:
        assert d == 1, 'uniform quantization only works for 1D data'

        # Pick clusters
        centers = torch.zeros((K, d), dtype=data.dtype, device=data.device)
        lin = torch.linspace(start=data_flt[:, 0].min(), 
                                end=data_flt[:, 0].max(), 
                                steps=K + 1, 
                                device=torch_dev)
        centers[:, 0] = (lin[:-1] + lin[1:]) / 2

        # indices
        idxs = torch.zeros_like(data_flt[:, 0], dtype=torch.int)
        b = centers[0, 0]
        if K > 1:
            m = centers[1, 0] - b
        else:
            m = 1
        idxs = torch.round((data_flt[:, 0] - b) / m).type(torch.int)
        idxs = torch.clamp(idxs, 0, K - 1)

    idxs = idxs.reshape(data.shape[:-1])

    return centers, idxs

def gen_grd(im_size: tuple, 
            fovs: Optional[tuple] = None) -> torch.Tensor:
    """
    Generates a grid of points given image size and FOVs

    Parameters:
    -----------
    im_size : tuple
        image dimensions
    fovs : tuple
        field of views, same size as im_size
    
    Returns:
    --------
    grd : torch.Tensor
        grid of points with shape (*im_size, len(im_size))
    """
    if fovs is None:
        fovs = (1,) * len(im_size)
    lins = [
        fovs[i] * torch.arange(-(im_size[i]//2), im_size[i]//2) / (im_size[i]) 
        for i in range(len(im_size))
        ]
    grds = torch.meshgrid(*lins, indexing='ij')
    grd = torch.cat(
        [g[..., None] for g in grds], dim=-1)
        
    return grd
    
def rotation_matrix(axis: torch.Tensor, 
                    theta: torch.Tensor) -> torch.Tensor:
    """
    Computes rotation matrices for a given axis and angle

    Parameters:
    -----------
    axis : torch.Tensor
        axis of rotation with shape (..., 3)
    theta : torch.Tensor
        angle of rotation in radians with shape (...)
    
    Returns:
    --------
    R : torch.Tensor
        rotation matrix with shape (..., 3, 3)
    """
    
    dev = axis.device
    axis = axis / torch.linalg.norm(axis, dim=-1)
    a = torch.cos(theta / 2.0)
    b = -axis[..., 0] * torch.sin(theta / 2.0)
    c = -axis[..., 1] * torch.sin(theta / 2.0)
    d = -axis[..., 2] * torch.sin(theta / 2.0)
    R = torch.zeros((*theta.shape, 3, 3), device=dev, dtype=torch.float32)
    R[..., 0, 0] = a * a + b * b - c * c - d * d
    R[..., 0, 1] = 2 * (b * c - a * d)
    R[..., 0, 2] = 2 * (b * d + a * c)
    R[..., 1, 0] = 2 * (b * c + a * d)
    R[..., 1, 1] = a * a + c * c - b * b - d * d
    R[..., 1, 2] = 2 * (c * d - a * b)
    R[..., 2, 0] = 2 * (b * d - a * c)
    R[..., 2, 1] = 2 * (c * d + a * b)
    R[..., 2, 2] = a * a + d * d - b * b - c * c
    return R

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

def np_to_torch(*args):
    """
    Converts numpy arrays to torch tensors,
    preserving device and dtype (uses CUPY)

    Parameters:
    -----------
    args : tuple
        numpy arrays to convert
    
    Returns:
    --------
    ret_args : tuple
        torch tensors
    """
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
    """
    Converts torch tensors to numpy arrays,
    preserving device and dtype (uses CUPY)

    Parameters:
    -----------
    args : tuple
        torch tensors to convert
    
    Returns:
    --------
    ret_args : tuple
        numpy arrays
    """
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
    """
    Iterator for batch processing

    Parameters:
    -----------
    total : int
        total number of elements
    batch_size : int
        batch size
    
    Returns:
    --------
    zip object
        iterator for batch processing
    """
    assert total > 0, f'batch_iterator called with {total} elements'
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])

def normalize(shifted, target, ofs=True, mag=False):
    """
    Assumes the following scaling/shifting offset:

    shifted = a * target + b

    solves for a, b and returns the corrected data

    Parameters:
    -----------
    shifted : np.ndarray
        data to be corrected
    target : np.ndarray
        reference data
    ofs : bool
        include b offset in the correction
    mag : bool
        use magnitude of data for correction
    
    Returns:
    --------
    np.ndarray
        corrected data
    """
    is_torch = torch.is_tensor(shifted)
    target = torch_to_np(target)
    shifted = torch_to_np(shifted)
    try:
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

        out = a * shifted + b
    except:
        print('Normalize Failed')
        out = shifted
    
    if is_torch:
        out = np_to_torch(out)
    return out
