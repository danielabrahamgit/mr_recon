import torch
import sigpy as sp

from mr_recon.fourier import fft
from mr_recon.algs import lin_solve
from mr_recon.dtypes import real_dtype
from mr_recon.utils import gen_grd, np_to_torch, torch_to_np, rotation_matrix

from typing import Optional, Tuple
from einops import rearrange, einsum
from sigpy.fourier import _get_oversamp_shape, _apodize, _scale_coord

def gen_source_vectors_rot_square(num_kerns: int,
                                  kern_size: tuple,
                                  dks: Optional[tuple] = None) -> torch.Tensor:
    """
    Make square kernels of size kern_size, then rotates them about center.
    
    Parameters:
    -----------
    num_kerns : int
        number of grappa kernels
    kern_size : tuple
        size of kernel. Example (5, 5) for a square kernel, (2, 5) for a rectangle, etc.
    dks : tuple
        specifies the spacing of the kernel in each dimension. If None, then all spacings are the same.
        
    Returns:
    --------
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
        ninputs = prod(kern_size)
    """
    
    # Make base kernel
    d = len(kern_size)
    if dks is None:
        dks = (1,) * d
    src_vecs_base = gen_grd(kern_size, kern_size).type(real_dtype).reshape((-1, d))
    for i in range(d):
        src_vecs_base[:, i] *= dks[i]
    src_vecs_base -= src_vecs_base.mean(dim=0)
    
    # Rotate
    if d == 2:
        thetas = torch.arange(num_kerns) * torch.pi / num_kerns
        axis = torch.tensor([0.0, 0.0, 1.0])[None,]
        R = rotation_matrix(axis=axis,
                            theta=thetas)[..., :2, :2]
    elif d == 3:
        thetas_x = torch.rand(num_kerns) * 2 * torch.pi
        thetas_y = torch.rand(num_kerns) * 2 * torch.pi
        thetas_z = torch.rand(num_kerns) * 2 * torch.pi
        R_x = rotation_matrix(axis=torch.tensor([1.0, 0.0, 0.0])[None,],
                              theta=thetas_x)
        R_y = rotation_matrix(axis=torch.tensor([0.0, 1.0, 0.0])[None,],
                              theta=thetas_y)
        R_z = rotation_matrix(axis=torch.tensor([0.0, 0.0, 1.0])[None,],
                              theta=thetas_z)
        R = einsum(R_z, R_y, R_x, 'N to1 to2, N to2 to3, N to3 to4 -> N to1 to4')
    source_vectors = einsum(R, src_vecs_base, 'N to ti, ninp ti -> N ninp to')
    return source_vectors

def gen_source_vectors_rot(num_kerns: int,
                           num_inputs: int,
                           ndim: int,
                           ofs: Optional[float] = 0.15,
                           line_width: Optional[float] = 2.0) -> torch.Tensor:
    """
    Generates a line of source points of width line_width with 
    some desitance ofs from the target points, and rotates about 
    the target to generate mulitple kernel vectors

    Parameters:
    -----------
    num_kerns : int
        number of grappa kernels
    num_inputs : int
        number of source point inputs
    ndim : int  
        number of dimensions (2 or 3 usually)
    ofs : float
        offset of closest source point to target point
    line_width : float
        width of line of source points
    
    Returns:
    --------
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
    """

    # Make line
    assert ndim >= 2
    source_vectors = torch.zeros(num_kerns, num_inputs, ndim)
    source_vectors[:, :, 0] = torch.linspace(-line_width / 2, line_width / 2, num_inputs)
    source_vectors[:, :, 1] = ofs

    # Rotate
    if ndim == 2:
        thetas = torch.arange(num_kerns) * 2 * torch.pi / num_kerns
        axis = torch.tensor([0.0, 0.0, 1.0])[None,]
        R = rotation_matrix(axis=axis,
                            theta=thetas)[..., :2, :2]
        source_vectors = einsum(R, source_vectors, 'N to ti, N ninp ti -> N ninp to')
    elif ndim == 3:
        thetas_x = torch.rand(num_kerns) * 2 * torch.pi
        thetas_y = torch.rand(num_kerns) * 2 * torch.pi
        thetas_z = torch.rand(num_kerns) * 2 * torch.pi
        R_x = rotation_matrix(axis=torch.tensor([1.0, 0.0, 0.0])[None,],
                              theta=thetas_x)
        R_y = rotation_matrix(axis=torch.tensor([0.0, 1.0, 0.0])[None,],
                              theta=thetas_y)
        R_z = rotation_matrix(axis=torch.tensor([0.0, 0.0, 1.0])[None,],
                              theta=thetas_z)
        R = einsum(R_z, R_y, R_x, 'N to1 to2, N to2 to3, N to3 to4 -> N to1 to4')
        source_vectors = einsum(R, source_vectors, 'N to ti, N ninp ti -> N ninp to')
    else:
        raise NotImplementedError
    
    return source_vectors

def gen_source_vectors_min_dist(num_kerns: int,
                                num_inputs: int,
                                ndim: int,
                                min_dist: Optional[float] = 0.5,
                                kern_width: Optional[float] = 5.0) -> torch.Tensor:
    """
    Generates random kernels 

    Parameters:
    -----------
    num_kerns : int
        number of grappa kernels
    num_inputs : int
        number of source point inputs
    ndim : int  
        number of dimensions (2 or 3 usually)
    min_dist : float
        minimum distance between source points
    kern_width : float
        size of kernel
    
    Returns:
    --------
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
    """

    # Initialize random points
    source_vectors = torch.rand(num_kerns, num_inputs, ndim) * kern_width - kern_width / 2

    # Ensure minimum distance
    for i in range(num_kerns):
        k = 0
        while True:
            diffs = source_vectors[i, :, None, :] - source_vectors[i, None, :, :]
            dists = diffs.norm(dim=-1)
            dists = dists + torch.eye(num_inputs) * 100
            if dists.min() > min_dist:
                break
            source_vectors[i] = torch.rand(num_inputs, ndim) * kern_width - kern_width / 2
            k += 1
            if k > 1000:
                print(f'Somethings wrong')

    return source_vectors

def gen_source_vectors_rand(num_kerns: int,
                            num_inputs: int,
                            ndim: int,
                            kern_width: Optional[float] = 5.0) -> torch.Tensor:
    """
    Generates random kernels 

    Parameters:
    -----------
    num_kerns : int
        number of grappa kernels
    num_inputs : int
        number of source point inputs
    ndim : int  
        number of dimensions (2 or 3 usually)
    kern_width : float
        size of kernel
    
    Returns:
    --------
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
    """

    source_vectors = torch.rand(num_kerns, num_inputs, ndim) * kern_width - kern_width / 2
    return source_vectors

def gen_source_vectors_circ(num_kerns: int,
                            num_inputs: int,
                            ndim: int,
                            diameter: Optional[float] = 5.0) -> torch.Tensor:
    """
    Generates random kernels 

    Parameters:
    -----------
    num_kerns : int
        number of grappa kernels
    num_inputs : int
        number of source point inputs
    ndim : int  
        number of dimensions (2 or 3 usually)
    diameter : float
        diameter of circles
    
    Returns:
    --------
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
    """
    raise not NotImplementedError
    return None

def rect_trj(cal_shape: tuple, 
             dk_buffer: Optional[int] = 2) -> torch.Tensor:
    """
    Creates a recti-linear trajectory.

    Parameters
    ----------
    cal_shape: tuple
        Shape of calibration, something like (nc, *spatial_dims)
    dk_buffer: int
        Edge points to remove from calibration

    Returns
    ---------
    trj_rect: torch.Tensor <float>
        The rect-linear coordinates with shape (ntrj, d)
    """

    d = len(cal_shape)
    rect_size = torch.tensor(cal_shape) - dk_buffer
    trj_rect = torch.zeros((*tuple(rect_size), d))
    
    for i in range(d):
        lin_1d = torch.arange(-rect_size[0]/2, rect_size[0]/2)
        tup = [None,]*d
        tup[i] = slice(None)
        trj_rect[..., i] = lin_1d[tuple(tup)]
    
    trj_rect = trj_rect.reshape((-1, d))

    return trj_rect

def grappa_AHA_AHb(img_cal: torch.Tensor, 
                   source_vectors: torch.Tensor,
                   width: Optional[int] = 6,
                   oversamp: Optional[float] = 1.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes AHA and AHb matrices for grappa kernel estimation using
    NUFFT interpolation on the calibration
    
    Parameters:
    -----------
    img_cal: torch.Tensor
        calibration image with shape (nc, *cal_size)
    source_vectors : torch.Tensor
        Coordinates of source relative to target with shape (nkerns, ninputs, d)
    width: int
            kaiser bessel kernel width
    oversamp: float
        kaiser bessel oversampling ratio
    
    Returns:
    ----------
    AHA: np.ndarray <complex>
        grappa calibration gram matrix with shape (nkerns, nc * ninputs, nc * ninputs) 
    AHb: np.ndarray <complex>
        grappa adjoint calibration times target poitns with shape (nkerns, nc * ninputs, nc)
    """

    # Consts
    device = img_cal.device
    beta = torch.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    n_coil = img_cal.shape[0]
    nkerns, num_inputs, d = source_vectors.shape
    assert device == source_vectors.device

    # Move to cupy
    img_cal_cp = torch_to_np(img_cal)
    kern_widths = 2 * source_vectors.cpu().reshape((-1, d)).abs().max(dim=0)[0]
    cal_size = (torch.tensor(img_cal.shape[1:]) - kern_widths).type(torch.int)
    target_vectors_cp = torch_to_np(gen_grd(cal_size, cal_size).reshape((-1, d)).to(device))
    source_vectors_cp = torch_to_np(source_vectors)
    dev = sp.get_device(img_cal_cp)

    # Prepare for KB interp
    with dev:

        # FFT part of NUFFT (copied from sigpy)
        os_shape = _get_oversamp_shape(img_cal_cp.shape, d, oversamp)
        output = img_cal_cp.copy()

        # Apodize
        _apodize(output, d, oversamp, width, beta)

        # Zero-pad
        output /= sp.util.prod(img_cal_cp.shape[-d:]) ** 0.5
        output = sp.util.resize(output, os_shape)

        # FFT
        ksp_cal_pre_KB = torch_to_np(fft(np_to_torch(output), dim=tuple(range(-d, 0)), norm=None))

    # KB interpolation
    with dev:
        
        # Target
        coord_trg = _scale_coord(target_vectors_cp, img_cal_cp.shape, oversamp)
        target = sp.interp.interpolate(
            ksp_cal_pre_KB, coord_trg, kernel="kaiser_bessel", width=width, param=beta
        )
        target /= width**d

        # Source
        source_vectors_cp = source_vectors_cp[:, None, ...] + target_vectors_cp[..., None, :]
        coord_src = _scale_coord(source_vectors_cp, img_cal_cp.shape, oversamp)
        source = sp.interp.interpolate(
            ksp_cal_pre_KB, coord_src, kernel="kaiser_bessel", width=width, param=beta
        )
        source /= width**d
    
    # Compute AHA and AHb
    source = rearrange(np_to_torch(source), 'nc N ncal ninp -> N ncal (nc ninp)')
    target = np_to_torch(target).T # ncal nc
    source_H = torch.moveaxis(source, -2, -1).conj()
    AHA = source_H @ source
    AHb = source_H @ target

    return AHA, AHb

def grappa_AHA_AHb_fast(img_cal: torch.Tensor, 
                        source_vectors: torch.Tensor, 
                        width: Optional[int] = 6, 
                        oversamp: Optional[float] = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A faster grappa kernel estimation algorithm from:
        Luo, T., Noll, D. C., Fessler, J. A., & Nielsen, J. (2019). 
        A GRAPPA algorithm for arbitrary 2D/3D non-Cartesian sampling trajectories with rapid calibration. 
        In Magnetic Resonance in Medicine. Wiley. https://doi.org/10.1002/mrm.27801
        
        Parameters
        ----------
        img_cal: torch.Tensor
            calibration image with shape (nc, *cal_size)
        source_vectors : ntorch.Tensor
            Coordinates of source relative to target with shape (nkerns, ninputs, d)
        width: int
            kaiser bessel kernel width
        oversamp: float
            kaiser bessel oversampling ratio

        Returns:
        ----------
        AHA: np.ndarray <complex>
            grappa calibration gram matrix with shape (nkerns, nc * ninputs, nc * ninputs) 
        AHb: np.ndarray <complex>
            grappa adjoint calibration times target poitns with shape (nkerns, nc * ninputs, nc)
        """
        
        # Consts
        device = img_cal.device
        beta = torch.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
        n_coil = img_cal.shape[0]
        nkerns, num_inputs, d = source_vectors.shape
        assert device == source_vectors.device

        # Move to cupy
        img_cal_cp = torch_to_np(img_cal)
        source_vectors_cp = torch_to_np(source_vectors)
        dev = sp.get_device(img_cal_cp)

        # Compute pairwise products of calibrations
        with dev:

            # FFT part of NUFFT (copied from sigpy)
            cross_img_cals = img_cal_cp[None, ...].conj() * img_cal_cp[:, None, ...]
            os_shape = _get_oversamp_shape(cross_img_cals.shape, d, oversamp)
            output = cross_img_cals.copy()

            # Apodize
            _apodize(output, d, oversamp, width, beta)

            # Zero-pad
            # output /= sp.util.prod(cross_img_cals.shape[-d:]) ** 0.5
            output = sp.util.resize(output, os_shape)

            # FFT
            cross_ksp_cals = torch_to_np(fft(np_to_torch(output), dim=tuple(range(-d, 0)), norm=None))

        # KB interpolation
        with dev:
            cross_orientations = -source_vectors_cp[:, None, :, :] + source_vectors_cp[:, :, None, :]
            coord_AHA = _scale_coord(cross_orientations, img_cal.shape, oversamp)
            AHA = sp.interp.interpolate(
                cross_ksp_cals, coord_AHA, kernel="kaiser_bessel", width=width, param=beta
            )
            AHA /= width**d

            # NUFFT based interpolation for AHb term
            coord_AHb = _scale_coord(-source_vectors_cp, img_cal.shape, oversamp)
            AHb = sp.interp.interpolate(
                cross_ksp_cals, coord_AHb, kernel="kaiser_bessel", width=width, param=beta
            )
            AHb /= width**d

        # Reshape
        AHb = rearrange(np_to_torch(AHb), 
                        'nco nci N ninp -> N (nci ninp) nco')
        AHA = rearrange(np_to_torch(AHA), 
                        'nco nci N ninpo ninpi -> N (nco ninpo) (nci ninpi)').conj()

        return AHA, AHb

def train_kernels(img_cal: torch.Tensor,
                  source_vectors: torch.Tensor,
                  lamda_tikonov: Optional[float] = 1e-3,
                  solver: Optional[str] = 'pinv',
                  fast_method: Optional[bool] = False,
                  return_errors: Optional[bool] = False) -> torch.Tensor:
    """
    Trains grappa kernels given calib image and source vectors

    Parameters:
    -----------
    img_cal : torch.Tensor
        calibration image with shape (ncoil, *cal_size)
    source_vectors : torch.Tensor
        vectors describing position of source relative to target with shape (nkerns, num_inputs, d)
    lamda_tikonov : float
        tikonov regularization parameter
    solver : str
        linear system solver from ['lstsq_torch', 'lstsq', 'pinv', 'inv']
    fast_method : bool
        toggles fast AHA AHb computation, only worth it for large calib
    return_errors : bool
        returns errors if True
    
    Returns:
    --------
    grappa_kernels : torch.Tensor
        GRAPPA kernels with shape (nkerns, ncoil, ncoil, num_inputs)
        maps num_input source points with ncoil channels to ncoil output target points
    """

    # Consts
    cal_size = img_cal.shape[1:]
    device = img_cal.device
    ncoil = img_cal.shape[0]
    nkerns, num_inputs, d = source_vectors.shape
    assert device == source_vectors.device

    # Compute AHA and AHb
    if fast_method:
        AHA, AHb = grappa_AHA_AHb_fast(img_cal, source_vectors)
    else:
        AHA, AHb = grappa_AHA_AHb(img_cal, source_vectors)
        
    # conds = torch.linalg.cond(AHA)# + lamda_tikonov * torch.eye(AHA.shape[-1], device=device))
    # print(conds.shape)
    # print(conds.min(), conds.max())
    # quit()
    
    # Solve
    AHA_old = AHA.clone()
    grappa_kernels = lin_solve(AHA, AHb, lamda=lamda_tikonov, solver=solver)
    if return_errors:
        errors = torch.mean((AHA_old @ grappa_kernels - AHb).abs().square(), dim=[-2,-1])
        grappa_kernels = rearrange(grappa_kernels, 'B (nci ninp) nco -> B nco nci ninp',
                                nci=ncoil, ninp=num_inputs)
        return grappa_kernels, errors
    grappa_kernels = rearrange(grappa_kernels, 'B (nci ninp) nco -> B nco nci ninp',
                               nci=ncoil, ninp=num_inputs)
    return grappa_kernels
