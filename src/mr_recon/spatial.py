import torch
import sigpy as sp

from typing import Optional
from torch.nn.functional import conv1d
from einops import rearrange
from mr_recon.fourier import fft, ifft
from mr_recon.utils import gen_grd, torch_to_np, np_to_torch, apply_window

def derivative(u: torch.Tensor,
               dim: Optional[int] = -1,
               method: Optional[str] = 'fourier',
               pre: Optional[bool] = False) -> torch.Tensor:
    """
    Compute the derivative of spatial tensor along dim.

    Parameters:
    -----------
    u : (torch.Tensor)
        The input tensor with shape (...)
    dim : (Optional[int])
        int specifying the dimensions to differentiate over
    method : (Optional[str])
        'conv' - use convolution to compute the derivative
        'diff' - use torch.diff to compute the derivative
        'fourier' - use the fourier transform to compute the derivative
    pre : (Optional[bool])
        Appends zero before kernel, or after depending on value of pre
    
    Returns:
    --------
    dff : (torch.Tensor)
        The derivative of u with shape (...)
    """

    if method == 'diff':
        if pre:
            prep = u[tuple([slice(None) if i != (dim % u.ndim) else slice(0, 1) for i in range(u.ndim)])]
            dff = torch.diff(u, dim=dim, prepend=prep)
        else:
            app = u[tuple([slice(None) if i != (dim % u.ndim) else slice(-1, None) for i in range(u.ndim)])]
            dff = torch.diff(u, dim=dim, append=app)
    elif method == 'conv':
        kern = torch.tensor([1, -8, 0, 8, -1]).to(u.device).type(u.dtype) / 12
        u_rs = u.moveaxis(dim, -1)
        og_shape = u_rs.shape
        u_rs = u_rs.reshape((-1, u_rs.shape[-1]))
        dff = conv1d(u_rs[:, None], kern[None, None], padding='same')[:, 0, :]
        dff = dff.reshape(og_shape)
        dff = dff.moveaxis(-1, dim).reshape(u.shape)
    elif method == 'fourier_conv':
        if pre:
            kern = torch.tensor([0,-1,1]).to(u.device).type(u.dtype)
        else:
            kern = torch.tensor([-1,1,0]).to(u.device).type(u.dtype)
        kern = torch_to_np(kern)
        with sp.get_device(kern):
            kern_rs = np_to_torch(sp.resize(kern, (u.shape[dim],)))
        kern_f = fft(kern_rs)
        tup = tuple([None if i != (dim % u.ndim) else slice(None) for i in range(u.ndim)])
        u_f = fft(u, dim=dim) * kern_f[tup]
        dff = ifft(u_f, dim=dim).real
    elif method == 'fourier':
        k = torch.arange(-(u.shape[dim]//2), u.shape[dim]//2, device=u.device) / u.shape[dim]
        tup = tuple([None if i != (dim % u.ndim) else slice(None) for i in range(u.ndim)])
        u_f = fft(u, dim=dim)
        u_f *= k[tup] * 1j * torch.pi
        dff = ifft(u_f, dim=dim)

    return dff

def apply_kern_1d(u: torch.Tensor,
                  kern: torch.Tensor,
                  dim: int) -> torch.Tensor:
    """
    Apply a kernel to a tensor along a single dimension.

    Parameters:
    -----------
    u : (torch.Tensor)
        The input tensor with shape (...)
    kern : (torch.Tensor)
        The kernel to apply with shape (n,)
    dim : (int)
        The dimension to apply the kernel over
    
    Returns:
    --------
    out : (torch.Tensor)
        The result of applying the kernel to u with shape (...)
    """
    u_rs = u.moveaxis(dim, -1)
    og_shape = u_rs.shape
    u_rs = u_rs.reshape((-1, u_rs.shape[-1]))
    out = conv1d(u_rs[:, None], kern[None, None], padding='same')[:, 0, :]
    out = out.reshape(og_shape)
    out = out.moveaxis(-1, dim).reshape(u.shape)
    return out

def laplacian(u: torch.Tensor,
              im_size: tuple,
              method: Optional[str] = 'fourier') -> torch.Tensor:
    """
    Compute the laplacian of spatial tensor.

    Parameters:
    -----------
    u : (torch.Tensor)
        The input tensor with shape (..., *im_size)
    im_size : (tuple)
        The size of the image
    method : (Optional[str])
        'conv' - use convolution to compute the laplacian
        'fourier' - use the fourier transform to compute the laplacian
        'der' - use the derivative to compute the laplacian
    
    Returns:
    --------
    lap : (torch.Tensor)
        The laplacian of u with shape (..., *im_size)
    """

    # Flatten batch dim
    u_flt = u.reshape((-1, *im_size))
    
    lap = u_flt * 0
    if method == 'fourier':
        # Apply laplacian operator in freq domain
        k = gen_grd(im_size).to(u.device) * 2 * torch.pi
        u_f = fft(u_flt, dim=tuple(range(-len(im_size), 0)))
        lap_f = -(torch.linalg.norm(k, dim=-1) ** 2) * u_f
        lap = ifft(lap_f, dim=tuple(range(-len(im_size), 0))).real
    elif 'conv' in method:
        # Compute the laplacian in the spatial domain
        if method == 'conv':
            method += '_h2'
        if 'h2' in method:
            kern = torch.tensor([1, -2, 1]).to(u_flt.device).type(u_flt.dtype)
        elif 'h4' in method:
            kern = torch.tensor([1, 16, -30, 16, -1]).to(u_flt.device).type(u_flt.dtype) / 12
        elif 'h6' in method:
            kern = torch.tensor([2, -27, 270, -490, 270, -27, 2]).to(u_flt.device).type(u_flt.dtype) / 180
        for i in range(-len(im_size), 0):
            lap += apply_kern_1d(u_flt, kern, dim=i)
    elif method == 'der':
        for i in range(-len(im_size), 0):
            lap += derivative(derivative(u_flt, dim=i, pre=True), dim=i, pre=False)
        lap = lap.real
    else:
        raise ValueError(f"Unknown method: {method}")

    # Restore batch dim
    return lap.reshape(u.shape)

def fourier_resize(x, new_shape, window='hamming'):

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
    X_rs = apply_window(np_to_torch(X_rs), ndim, window)

    # IFFT
    x_rs = ifft(X_rs, dim=tuple(range(-ndim, 0)))

    # Convert to original
    if torch.is_tensor(x):
        return x_rs
    else:
        return torch_to_np(x_rs)

def spatial_resize(x: torch.Tensor,
                   im_size: tuple, 
                   method: Optional[str] = 'bilinear') -> torch.Tensor:
    """
    Resize a spatial tensor to a new spatial size.

    Parameters:
    -----------
    x : (torch.Tensor)
        The input tensor with shape (..., *inp_im_size)
    im_size : (tuple)
        The size of the image to resize to
    method : (Optional[str])
        The method to use for resizing, options are:
            - 'bilinear': linear interpolation (default)
            - 'bicubic': cubic interpolation
            - 'nearest': nearest sample interpolation
            - 'fourier': Fourier interpolation
    
    Returns:
    --------
    x_rs : (torch.Tensor)
        The resized tensor with shape (..., *im_size)
    """

    # Keep orig batch dims, then flatten
    orig_batch = x.shape[:-len(im_size)]
    orig_im_size = x.shape[-len(im_size):]
    x_flt = x.reshape((-1, *orig_im_size))

    if method == 'fourier':
        x_rs_flt = fourier_resize(x_flt, im_size)
    else:
        grd = 2 * gen_grd(im_size).to(x.device)
        grd -= grd.min()
        grd *= 2 / grd.max()
        grd -= 1
        grd = grd.flip(-1)
        grd = grd[None,]
        if torch.is_complex(x):
            x_rs_flt = torch.nn.functional.grid_sample(x_flt.real[None,], grd, align_corners=True, mode=method) + \
                    1j * torch.nn.functional.grid_sample(x_flt.imag[None,], grd, align_corners=True, mode=method)
        else:
            x_rs_flt = torch.nn.functional.grid_sample(x_flt[None,], grd, align_corners=True, mode=method)
        x_rs_flt = x_rs_flt[0]
    
    x_rs = x_rs_flt.reshape((*orig_batch, *im_size))
    return x_rs