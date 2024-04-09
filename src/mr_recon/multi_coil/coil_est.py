import time
import torch
import sigpy as sp

from tqdm import tqdm
from typing import Optional, Tuple
from einops import rearrange, einsum
from torchkbnufft import KbNufftAdjoint
from mr_recon.algs import power_method_matrix
from mr_recon.fourier import ifft
from mr_recon.utils import torch_to_np, np_to_torch

# TODO give batch dims (useful for multi-slice)
def csm_from_espirit(ksp_cal: torch.Tensor,
                     im_size: tuple,
                     thresh: Optional[float] = 0.02,
                     kernel_width: Optional[int] = 6,
                     crp: Optional[float] = 0.95,
                     max_iter: Optional[int] = 100,
                     verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copy of sigpy implementation of ESPIRiT calibration, but in torch:
    Martin Uecker, ... ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI

    Parameters:
    -----------
    ksp_cal : torch.Tensor
        Calibration k-space data with shape (ncoil, *cal_size)
    im_size : tuple
        output image size
    thresh : float
        threshold for SVD nullspace
    kernel_width : int
        width of calibration kernel
    crp : float
        output mask based on copping eignevalues
    max_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (ncoil, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    # Consts
    img_ndim = len(im_size)
    num_coils = ksp_cal.shape[0]
    device = ksp_cal.device

    # TODO torch this part
    # Get calibration matrix.
    # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
    ksp_cal_sp = torch_to_np(ksp_cal)
    dev = sp.get_device(ksp_cal_sp)
    with dev:
        mat = sp.array_to_blocks(
            ksp_cal_sp, [kernel_width] * img_ndim, [1] * img_ndim)
        mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
        mat = mat.transpose([1, 0, 2])
        mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])
    mat = np_to_torch(mat)

    # Perform SVD on calibration matrix
    if verbose:
        print('Computing SVD on calibration matrix: ', end='')
        start = time.perf_counter()
    _, S, VH = torch.linalg.svd(mat, full_matrices=False)
    VH = VH[S > thresh * S.max(), :]
    if verbose:
        end = time.perf_counter()
        print(f'{end - start:.3f}s')

    # Get kernels
    num_kernels = len(VH)
    kernels = VH.reshape(
        [num_kernels, num_coils] + [kernel_width] * img_ndim)

    # Get covariance matrix in image domain
    AHA = torch.zeros(im_size + (num_coils, num_coils), 
                        dtype=ksp_cal.dtype, device=device)
    for kernel in tqdm(kernels, 'Computing covariance matrix', disable=not verbose):
        img_kernel = ifft(kernel, oshape=(num_coils, *im_size),
                                dim=tuple(range(-img_ndim, 0)))
        aH = rearrange(img_kernel, 'nc ... -> ... nc 1')
        a = aH.swapaxes(-1, -2).conj()
        AHA += aH @ a
    AHA *= (torch.prod(torch.tensor(im_size)).item() / kernel_width**img_ndim)
    
    # Get eigenvalues and eigenvectors
    mps, eigen_vals = power_method_matrix(AHA, num_iter=max_iter, verbose=verbose)
    
    # Phase relative to first map
    mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-8))
    mps *= eigen_vals > crp

    return mps, eigen_vals

def csm_from_kernels(grappa_kernels: torch.Tensor,
                     source_vectors: torch.Tensor,
                     im_size: tuple,
                     crp: Optional[float] = 0.95,
                     max_iter: Optional[int] = 100,
                     verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates coil sensitivty maps from grappa kernels

    Parameters:
    -----------
    grappa_kernels : torch.Tensor
        GRAPPA kernels with shape (nkerns, ncoil, ncoil, num_inputs)
        maps num_input source points with ncoil channels to ncoil output target points
    source_vectors : torch.Tensor
        vectors describing position of source relative to target with shape (nkerns, num_inputs, d)
    im_size : tuple
        output image size
    crp : float
        crops based on eignevalues
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (ncoil, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    # Compute image covariance and do power method
    BHB = calc_image_covariance_kernels(grappa_kernels, source_vectors, im_size, verbose=verbose)
    mps, eigen_vals = power_method_matrix(BHB, num_iter=max_iter, verbose=verbose)

    # Phase relative to first map
    mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-8))
    mps *= eigen_vals > crp

    return mps, eigen_vals

def calc_image_covariance_kernels(grappa_kernels: torch.Tensor,
                                  source_vectors: torch.Tensor,
                                  im_size: tuple,
                                  coil_batch: Optional[int] = None,
                                  sigpy_nufft: Optional[bool] = True,
                                  verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Calculates B^HB matrix -- see writeup for details

    Parameters:
    -----------
    grappa_kernels : torch.Tensor
        GRAPPA kernels with shape (nkerns, ncoil, ncoil, num_inputs)
        maps num_input source points with ncoil channels to ncoil output target points
    source_vectors : torch.Tensor
        vectors describing position of source relative to target with shape (nkerns, num_inputs, d)
    im_size : tuple
        output image size
    coil_batch : int
        number of coils to process at once
    sigpy_nufft : bool
        use sigpy nufft instead of torchkbnufft
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    BHB : torch.Tensor
        image covariance kernels with shape (ncoil, ncoil, *im_size)
    """

    # Consts
    device = grappa_kernels.device
    nkerns, ncoil, _, num_inputs = grappa_kernels.shape
    d = source_vectors.shape[-1]
    assert nkerns == source_vectors.shape[0]
    assert num_inputs == source_vectors.shape[1]
    assert device == source_vectors.device
    if coil_batch is None or coil_batch > ncoil ** 2:
        coil_batch = ncoil

    # Make cross terms
    grappa_kerns_rs = rearrange(grappa_kernels, 'N nco nci ninp -> nco nci N ninp')
    source_vectors_cross = source_vectors[:, :, None, :] - source_vectors[:, None, :, :] # nkerns ninp ninp d
    grappa_kerns_cross = einsum(grappa_kerns_rs.conj(), grappa_kerns_rs, 
                        'nc nci B ninp, nc nci2 B ninp2 -> nci nci2 B ninp ninp2')
    grappa_kerns_rs_conj = rearrange(grappa_kerns_rs, 'nco nci N ninp -> nci nco N ninp').clone()
    grappa_kerns_rs_conj.imag *= -1

    # Flatten coils and rescale
    scale = (torch.prod(torch.tensor(im_size)).item() ** 0.5) / nkerns
    grappa_kerns_rs = rearrange(grappa_kerns_rs, 'nco nci N ninp -> (nco nci) N ninp')
    grappa_kerns_rs *= scale
    grappa_kerns_rs_conj = rearrange(grappa_kerns_rs_conj, 'nci nco N ninp -> (nci nco) N ninp')
    grappa_kerns_rs_conj *= scale
    grappa_kerns_cross = rearrange(grappa_kerns_cross, 'nci nci2 N ninp ninp2 -> (nci nci2) N ninp ninp2')
    grappa_kerns_cross *= scale

    # Using sigpy for adjoint nufft
    if sigpy_nufft:
        grappa_kerns_cross_sp = torch_to_np(grappa_kerns_cross)
        grappa_kerns_rs_sp = torch_to_np(grappa_kerns_rs)
        grappa_kerns_rs_conj_sp = torch_to_np(grappa_kerns_rs_conj)
        source_vectors_cross_sp = torch_to_np(source_vectors_cross)
        source_vectors_sp = torch_to_np(source_vectors)
        dev = sp.get_device(grappa_kerns_cross_sp)
        xp = dev.xp

        # Build BHB matrix
        with dev:
            BHB = xp.zeros((ncoil ** 2, *im_size), dtype=grappa_kerns_rs_sp.dtype)
            for c1 in tqdm(range(0, ncoil ** 2, coil_batch), 'Computing Covariance Matrix', disable=not verbose):
                c2 = min(ncoil ** 2, c1 + coil_batch)
                oshape = (c2 - c1, *im_size)
                BHB[c1:c2] += sp.nufft_adjoint(grappa_kerns_cross_sp[c1:c2], source_vectors_cross_sp, oshape, width=6)
                BHB[c1:c2] += -sp.nufft_adjoint(grappa_kerns_rs_sp[c1:c2], -source_vectors_sp, oshape, width=6)
                BHB[c1:c2] += -sp.nufft_adjoint(grappa_kerns_rs_conj_sp[c1:c2], source_vectors_sp, oshape, width=6)
            BHB = rearrange(BHB, '(nc nci) ... -> ... nc nci',
                            nc=ncoil, nci=ncoil)
            BHB = np_to_torch(BHB)
    # TorchKBNUFFT for adjoint
    else:
        im_size_arr = torch.tensor(im_size).to(device)
        source_vectors = torch.pi * source_vectors / (im_size_arr / 2)
        source_vectors_cross = torch.pi * source_vectors_cross / (im_size_arr / 2)

        # Build BHB matrix
        BHB = torch.zeros((ncoil * ncoil, *im_size), dtype=grappa_kerns_rs.dtype, device=device)
        kbn = KbNufftAdjoint(im_size, dtype=grappa_kerns_rs.dtype, device=device)
        source_vectors = rearrange(source_vectors, 'N ninp d -> d (N ninp)')
        source_vectors_cross = rearrange(source_vectors_cross, 'N ninp ninp2 d -> d (N ninp ninp2)')
        grappa_kerns_rs = rearrange(grappa_kerns_rs, 'C N ninp -> 1 C (N ninp)')
        grappa_kerns_rs_conj = rearrange(grappa_kerns_rs_conj, 'C N ninp -> 1 C (N ninp)')
        grappa_kerns_cross = rearrange(grappa_kerns_cross, 'C N ninp ninp2 -> 1 C (N ninp ninp2)')
        for c1 in tqdm(range(0, ncoil ** 2, coil_batch), 'Computing Covariance Matrix', disable=not verbose):
            c2 = min(ncoil ** 2, c1 + coil_batch)
            BHB[c1:c2] += kbn(grappa_kerns_cross[:, c1:c2], source_vectors_cross)[0]
            BHB[c1:c2] += -kbn(grappa_kerns_rs[:, c1:c2], -source_vectors)[0]
            BHB[c1:c2] += -kbn(grappa_kerns_rs_conj[:, c1:c2], source_vectors)[0]
        BHB = rearrange(BHB, '(nc nci) ... -> ... nc nci',
                        nc=ncoil, nci=ncoil)

    return BHB
