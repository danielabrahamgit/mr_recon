from typing import Optional, Literal
from warnings import warn
from einops import rearrange

import numpy as np
import sigpy as sp
import sigpy.mri as mri
import torch
import torch.fft as fft
from torchkbnufft import (
    KbNufft,
    KbNufftAdjoint,
    ToepNufft,
    calc_toeplitz_kernel,
)

from igrog.recon.torch_algs import conjugate_gradient

__all__ = [
    'truncate_trj_ksp',
    'to_cartesian',
    'extract_acs_region',
    'synth_cal',
    'get_mps_kgrid_toep_pcg',
]

def sp_fft(x, dim=None):
    """Matches Sigpy's fft, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x

def sp_ifft(x, dim=None, norm=None):
    """Matches Sigpy's fft adjoint, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.ifftn(x, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x

def convert_trj(trj, from_type, to_type, im_size = None):
    if (from_type == 'sigpy' or to_type == 'sigpy') and im_size == None:
        raise ValueError('Must specify im_size if converting to/from sigpy.')
    if from_type == to_type:
        return trj
    # Convert to mat first
    if from_type == 'mat':
        pass
    elif from_type == 'tkbn':
        trj = trj / (2*np.pi)
        trj = rearrange(trj, '... d k -> ... k d')
    elif from_type == 'sigpy':
        for i, N in enumerate(im_size):
            trj[..., i] = trj[..., i] / N
    else:
        raise ValueError(f'Invalid from_type: {from_type}')
    # Convert to whatever else
    if to_type == 'mat':
        pass
    elif to_type == 'tkbn':
        trj = trj * 2 * np.pi
        trj = rearrange(trj, '... k d -> ... d k')
    elif to_type == 'sigpy':
        for i, N in enumerate(im_size):
            trj[..., i] = trj[..., i] * N
    else:
        raise ValueError(f'Invalid to_type: {to_type}')
    return trj

def truncate_trj_ksp(trj, ksp, max_k, dcf: Optional[np.ndarray] = None):
    """Truncate trajectory and ksp (useful for e.g. only using minimum number of trj points needed
    to generate ACS region)

    K: readout dimension
    D: spatial dimension

    trj: [S... K D]
    ksp: [C S... K]
    trj and ksp should match on the S... dimensions
    ksp may have an arbitrary number of dimensions prepended
    (e.g. coil dimensions)

    Return:
    -------
    trj_truncated: [N D]
    ksp_truncated: [C... N]
    """
    mask = np.all(np.abs(trj) <= max_k, axis=-1) # [S... K]
    trj_truncated = trj[mask, :]
    ksp_truncated = ksp[..., mask]
    if dcf is not None:
        dcf_truncated = dcf[mask]
        return trj_truncated, ksp_truncated, dcf_truncated
    return trj_truncated, ksp_truncated


def to_cartesian(trj, ksp, im_size, dcf: Optional[np.ndarray] = None):
    """Convert non-cartesian kspace to cartesian kspace
    for e.g coil estimation with espirit
    trj: [R T K D] with sigpy scaling (i.e. [-N/2, N/2])
    ksp: [C R T K] kspace points
    dcf: [R T K]: density compensation (Not recommended)
    im_size: Tuple of integer: Desired image size
    """
    C = ksp.shape[0] # Number of coils
    D = trj.shape[-1] # Dimension of problem
    assert isinstance(im_size, int) or len(im_size) == D, \
        f'im_size must be an int or array of length {D}'
    if isinstance(im_size, int):
        im_size = [im_size] * D
    device = sp.get_device(trj)
    xp = device.xp
    with device:
        # iNUFFT to get "image"
        # Keep coil dim
        # Average across all groups and time points
        trj = xp.broadcast_to(trj, shape=(*ksp.shape[1:], D))
        if dcf is not None:
            ksp = ksp * dcf
        img = sp.nufft_adjoint(ksp, trj, oshape=(C, *im_size))

        # Fourier transform again, but cartesian
        Fimg = sp.fft(img, axes=tuple(range(-D, 0)), norm='ortho')
    return Fimg

def extract_acs_region(trj, ksp, dcf, im_size, acs_size: int):
    """
    trj: [R T K D] with sigpy scaling (i.e. [-N/2, N/2])
    ksp: [C R T K] kspace points
    dcf: [R T K]
    im_size: Tuple of integer:
    acs_size: int or tuple of integers
    batch_size: int if ksp data is too large
    """
    assert isinstance(acs_size, int), f'acs_size must be an int'
    D = trj.shape[-1]
    device = sp.get_device(trj)
    xp = device.xp
    with device:
        Fimg = to_cartesian(trj, ksp, dcf, im_size)
        # Extract ACS region from center of kspace
        # Slicing with an array requires it to be a tuple
        acs_slc = tuple(
            slice(im_size[i]//2 - acs_size//2, im_size[i]//2 + acs_size//2)
            for i in range(len(im_size))
        )
        full_slc = (slice(None),) * (len(Fimg.shape) - D) + acs_slc
        Fimg_acs = Fimg[full_slc]
    return Fimg_acs

def synth_cal(trj, ksp, acs_size: int,
              method: Literal['inverse', 'adjoint'] = 'adjoint',
              dcf: Optional[np.ndarray] = None,
              device='cpu',
              toeplitz: bool = False
):
                  # cal_size: Optional[int] = 32,
                  # coil_batch_size: Optional[int] = 1) -> np.ndarray:
        """
        Synthesizes calibration region from calibration data.

        S... = arbitrary shape (e.g. [npe, ntr])
        C    = coil dimension
        K    = readout dimension i.e. nro
        D    = Spatial dimension


        Parameters
        ----------
        trj : :np.ndarray <float>
            [S... K D] in [-N/2, N/2]
        ksp : np.ndarray <complex>
            k-space raw data with shape [C S... K]
        acs_size : int
            width of calibration along each dimension (only square for now)
        dcf : Optional[np.ndarray] <float>
            Optional density compensation function [S... K]

        Returns
        ---------
        ksp_cal : np.ndarray <complex>
            calibration signal with shape (nc, acs_size, acs_size, (acs_size))
        """

        # Consts
        D = trj.shape[-1] # Number of spatial dimensions
        C = ksp.shape[0]  # Number of coils
        if not (D == 2 or D == 3):
            warn('Image is not 2D or 3D - are you sure this is correct?')
        cal_size = (acs_size,) * D

        # Use all points within limits
        if dcf is not None:
            trj, ksp, dcf = truncate_trj_ksp(trj, ksp, max_k=acs_size/2, dcf=dcf)
        else:
            trj, ksp = truncate_trj_ksp(trj, ksp, max_k=acs_size/2)

        # Move to pytorch
        omega = convert_trj(trj, 'sigpy', 'tkbn', im_size=cal_size)
        omega = torch.as_tensor(omega)
        ksp = torch.as_tensor(ksp)
        if dcf is not None:
            dcf = torch.as_tensor(dcf)
            ksp = ksp * dcf

        # Adjoint NUFFT + FFT back to kspace
        if method == 'adjoint':
            nufft_adj = KbNufftAdjoint(cal_size).to(device)
            omega = omega.to(device)
            ksp = ksp.to(device)
            img_cal = nufft_adj(data=ksp[None, ...], omega=omega, norm='ortho')[0]
        elif method == 'inverse':
            img_cal = inverse_nufft_pcg(omega, ksp, cal_size, dcf=dcf, device=device, toeplitz=toeplitz)
            img_cal = img_cal[0] # remove batch dimension

        ksp_cal = sp_fft(img_cal, dim=tuple(range(-D, 0)))

        return ksp_cal.detach().cpu().numpy()


def inverse_nufft_pcg(omega, ksp, cal_size,
                      dcf=None,
                      device='cpu',
                      toeplitz: bool = False):
    C = ksp.shape[0] # Coil dim
    nufft_adj = KbNufftAdjoint(cal_size, device=device).to(device)
    if toeplitz:
        nufft = KbNufft(cal_size, device=device).to(device)
        def A_op(x: torch.Tensor):
            """AHA operation
            x: [1]
            """
            with torch.cuda.device(device):
                y = nufft(x, omega=omega, norm='ortho')
                if dcf is not None:
                    y = y * dcf
                out = nufft_adj(y, omega=omega, norm='ortho')
            return out
    else:
        toep = ToepNufft().to(device)
        kernel = calc_toeplitz_kernel(omega, im_size=cal_size, norm='ortho').to(device)
        def A_op(x: torch.Tensor):
            return toep(x, kernel)

    y = ksp[None, ...].to(device)
    omega = omega.to(device)
    if dcf is not None:
        dcf = dcf.to(device)
    #omega = omega[None, ...]
    #dummy_smaps = torch.ones((C, *cal_size), dtype=torch.complex64).to(device)
    with torch.cuda.device(device):
        x_init = nufft_adj(y, omega=omega, norm='ortho')
    img_cal = conjugate_gradient(AHA=A_op,
                                 AHb=x_init)
    return img_cal


def get_mps_kgrid_toep_pcg(
        ksp, trj, im_size,
        calib_width: int = 24,
        kernel_width: int = 6,
        crop: float = 0.8,
        thresh: float = 0.05,
        device_idx: int = -1,
):
    sp_device = sp.Device(device_idx)
    th_device = torch.device(f'cuda:{device_idx}' if device_idx >= 0 else 'cpu')
    dcf = mri.pipe_menon_dcf(trj, im_size, device=sp_device)
    dcf /= np.linalg.norm(dcf)

    # Solve for ACS at reduced size for speed
    padded_calib_width = calib_width + kernel_width // 2 + 1
    kgrid = synth_cal(trj, ksp, padded_calib_width,
                      dcf=dcf,
                      method='inverse',
                      device=th_device,
                      toeplitz=True)

    # Pad to full image size
    C = kgrid.shape[0]
    kgrid_pad = sp.resize(kgrid, (C, *im_size))

    mps = mri.app.EspiritCalib(
        kgrid_pad,
        calib_width=calib_width,
        kernel_width=kernel_width,
        crop=crop,
        thresh=thresh,
        device=sp_device
    ).run()

    return mps, kgrid