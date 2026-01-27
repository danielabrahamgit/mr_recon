import time
import torch

from tqdm import tqdm
from typing import Optional, Tuple
from einops import rearrange, einsum
from mr_recon.dtypes import complex_dtype
from mr_recon.algs import power_method_matrix, lobpcg_operator
from mr_recon.utils import batch_iterator, tqdm_batch_iterator
from mr_recon.fourier import ifft, NUFFT, sigpy_nufft
from mr_recon.block import array_to_blocks
from mr_recon.multi_coil.grappa_utils import gen_source_vectors_rand, train_kernels, gen_source_vectors_rot_square

def _espirit_batch_sizes(
    kernel_batch_size: Optional[int],
    ndim: int,
    cpu_last_part: bool,
    Nc: int,
):
    """
    Heuristic batch sizes for ESPRIT when using in 2D or 3D
    """
    bs_kern = 1
    bs_aha = 1
    bs_fft = 1
    
    if ndim == 2:
        if kernel_batch_size is not None:
            bs_kern = kernel_batch_size
        # don't batch AHA and FFT, should fit on 1 GPU
        bs_aha = Nc
        bs_fft = Nc
    elif ndim == 3:
        bs_kern = 1
        if cpu_last_part:
            bs_aha = 16
            bs_fft = 10
        else:
            bs_aha = 1
            bs_fft = 5
    
    batch_aha = (bs_aha < Nc)
    batch_fft = (bs_fft < Nc)

    return bs_kern, bs_aha, bs_fft, batch_aha, batch_fft

def csm_from_espirit(ksp_cal: torch.Tensor,
                     im_size: tuple,
                     thresh: Optional[float] = 0.02,
                     kernel_width: Optional[int] = 6,
                     crp: Optional[float] = None,
                     sets_of_maps: Optional[int] = 1,
                     max_iter: Optional[int] = 300,
                     lobpcg_iter: Optional[int] = None,
                     cpu_last_part: Optional[bool] = False,
                     kernel_batch_size: Optional[int] = None,
                     verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copy of sigpy implementation of ESPIRiT calibration, but in torch:
    Martin Uecker, ... ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI

    Parameters:
    -----------
    ksp_cal : torch.Tensor
        Calibration k-space data with shape (Nc, *cal_size)
    im_size : tuple
        output image size
    thresh : float
        threshold for SVD nullspace
    kernel_width : int
        width of calibration kernel
    crp : float
        output mask based on copping eignevalues
    sets_of_maps : int
        number of sets of maps to compute
    max_iter : int
        number of iterations to run power method
    lobpcg_iter : int
        number of iterations to run lobpcg
        if given, uses lobpcg instead of svd for first part
    cpu_last_part : bool
        if True, moves AHA and after computations to CPU (for 3D problems)
    kernel_batch_size : int
        batch size for computing AHA over ESPIRIT kernels (can be >1 if 2D problems)
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (Nc, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    # Consts
    img_ndim = len(im_size)
    Nc = ksp_cal.shape[0]
    device = ksp_cal.device

    # batching things
    bs_kern, bs_aha, bs_fft, batch_aha, batch_fft = _espirit_batch_sizes(
        kernel_batch_size, img_ndim, cpu_last_part, Nc
    )

    # Get calibration matrix.: [Nc] + num_blks + [kernel_width] * img_ndim
    mat = array_to_blocks(
        ksp_cal, [kernel_width] * img_ndim, [1] * img_ndim
    ).reshape(Nc, -1, kernel_width**img_ndim)
    mat = mat.permute(1, 0, 2).reshape(-1, Nc * kernel_width**img_ndim)

    # Perform SVD on calibration matrix
    if verbose:
        print('[ESIRIT]: Computing SVD on calibration matrix: ', end='')
        start = time.perf_counter()

    if lobpcg_iter is not None:
        AHA = mat.H @ mat
        A_op = lambda x : AHA @ x
        num_eigen = min(AHA.shape[1], 5000)
        X = torch.randn((AHA.shape[0], num_eigen), dtype=AHA.dtype, device=AHA.device)
        evals, evecs = lobpcg_operator(A_op, X, maxiter=lobpcg_iter)
        S = evals.abs() ** 0.5
        VH = evecs.H
    else:
        _, S, VH = torch.linalg.svd(mat, full_matrices=False)
    VH = VH[S > thresh * S.max(), :]

    if verbose:
        end = time.perf_counter()
        print(f'{end - start:.3f}s')

    # Move to CPU if last part is on CPU, for large 3D problems
    if cpu_last_part:
        old_dev = device
        device = torch.device('cpu')
    else:
        old_dev = device
    
    # ------- Compute covariance matrix -------   
    num_kernels = len(VH)
    kernels = VH.reshape([num_kernels, Nc] + [kernel_width] * img_ndim)
    AHA = torch.zeros(im_size + (Nc, Nc), dtype=ksp_cal.dtype, device=device)
    kernels = kernels.to(device)

    tqdm_k_kwargs = {"desc": "[ESPIRIT]: covariance matrix", "disable": not verbose}
    tqdm_aha_kwargs = {"desc": "[ESPIRIT AHA]: Matmul batches", "disable": not verbose, "leave": False}
    for ki, ke in tqdm_batch_iterator(num_kernels, batch_size=bs_kern, **tqdm_k_kwargs):
        kb = ke - ki
        # iFFT kernel
        if batch_fft:
            aH = torch.zeros((kb, *im_size, Nc, 1), dtype=kernels.dtype, device=old_dev)
            for ci, cl in batch_iterator(Nc, bs_fft):
                ah_ = ifft(kernels[ki:ke, ci:cl].to(old_dev), oshape=(kb, cl-ci, *im_size), dim=tuple(range(-img_ndim, 0)))
                aH[..., ci:cl, :] = rearrange(ah_, 'nk nc ... -> nk ... nc 1')
            aH = aH.to(device)
        else:
            aH = ifft(kernels[ki:ke], oshape=(kb, Nc, *im_size), dim=tuple(range(-img_ndim, 0))).moveaxis(1, -1)[..., None]

        # Add to AHA
        if batch_aha:
            for c1, c2 in tqdm_batch_iterator(Nc, batch_size=bs_aha, **tqdm_aha_kwargs):
                AHA[..., c1:c2, :] += ((aH[..., c1:c2, :] @ aH.mH)).sum(dim=0)
        else:
            AHA += ((aH @ aH.mH)).sum(dim=0)

    AHA *= (torch.prod(torch.tensor(im_size)).item() / kernel_width**img_ndim)

    # Get eigenvalues and eigenvectors
    mps_all = []
    evals_all = []
    for i in range(sets_of_maps):
        
        # power iterations
        mps, eigen_vals = power_method_matrix(AHA, num_iter=max_iter, verbose=verbose)
        
        # Update AHA
        if sets_of_maps > 1:
            AHA -= einsum(mps * eigen_vals, mps.conj(), 'Cl ..., Cr ... -> ... Cl Cr')
        mps_all.append(mps)
        evals_all.append(eigen_vals)
        
    if sets_of_maps == 1:
        mps = mps_all[0]
        eigen_vals = evals_all[0]
    else:
        mps = torch.stack(mps_all, dim=1) # C S *im_size
        eigen_vals = torch.stack(evals_all, dim=0) # S *im_size
    
    # Phase relative to first map and crop
    mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-12))
    if crp:
        mps *= eigen_vals > crp

    return mps, eigen_vals


def csm_from_grappa(ksp_cal: torch.Tensor,
                    im_size: tuple,
                    num_kerns: Optional[int] = 100,
                    num_src: Optional[int] = 25,
                    kernel_width: Optional[int] = 6,
                    lamda_tikonov: Optional[float] = 0.0,
                    sets_of_maps: Optional[int] = 1,
                    crp: Optional[float] = None,
                    max_iter: Optional[int] = 100,
                    min_eigen_value: Optional[bool] = True,
                    verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trains grappa kernels and then calls csm_from_kernels.

    Parameters:
    -----------
    ksp_cal : torch.Tensor
        Calibration k-space data with shape (ncoil, *cal_size)
    im_size : tuple
        output image size
    num_kerns : int
        number of kernels to train
    num_src : int
        number of source points to use in training
    kernel_width : int
        width of calibration kernel
    sets_of_maps : int
        number of sets of maps to compute
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
    torch_dev = ksp_cal.device
    img_cal = ifft(ksp_cal, dim=list(range(-len(im_size), 0)))

    # Generate random source vectors
    # src_vecs = gen_source_vectors_min_dist(num_kerns=num_kerns, 
    #                                        num_inputs=num_src, 
    #                                        ndim=len(im_size), 
    #                                        min_dist=0.2,
    #                                        kern_width=kernel_width)
    if type(kernel_width) == tuple:
        src_vecs = gen_source_vectors_rot_square(num_kerns=num_kerns,
                                                kern_size=kernel_width,
                                                dks=(1.0,)*2)
    else:
        src_vecs = gen_source_vectors_rand(num_kerns=num_kerns, 
                                        num_inputs=num_src, 
                                        ndim=len(im_size), 
                                        kern_width=kernel_width)
        # src_vecs = gen_source_vectors_rot(num_kerns=num_kerns, 
        #                                 num_inputs=num_src, 
        #                                 ndim=len(im_size), 
        #                                 ofs=0.15, 
        #                                 line_width=kernel_width)

    src_vecs = src_vecs.to(torch_dev)
    batch_size = 50
    kerns = None
    for n1 in tqdm(range(0, num_kerns, batch_size), 'Training Kernels'):
        n2 = min(num_kerns, n1 + batch_size)
        kerns_batch = train_kernels(img_cal, src_vecs[n1:n2], 
                                    fast_method=False, 
                                    solver='solve',
                                    lamda_tikonov=lamda_tikonov).type(complex_dtype)
        if kerns is None:
            kerns = kerns_batch
        else:
            kerns = torch.cat((kerns, kerns_batch), dim=0)

    return csm_from_kernels(kerns, src_vecs, im_size, crp, max_iter, sets_of_maps, min_eigen_value, verbose)

def csm_from_kernels(grappa_kernels: torch.Tensor,
                     source_vectors: torch.Tensor,
                     im_size: tuple,
                     crp: Optional[float] = None,
                     max_iter: Optional[int] = 100,
                     sets_of_maps: Optional[int] = 1,
                     min_eigen_value: Optional[bool] = True,
                     verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates coil sensitivty maps from trained grappa kernels

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
    sets_of_maps : int
        number of sets of maps to compute
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (ncoil, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    # Compute image covariance kernels
    BHB = calc_image_covariance_kernels(grappa_kernels, source_vectors, im_size, verbose=verbose)
    if min_eigen_value:
        BHB += torch.eye(BHB.shape[-1], device=BHB.device, dtype=BHB.dtype) 

        # Min eigen from max
        _, eig_vals_max = power_method_matrix(BHB, num_iter=max_iter, verbose=verbose)
        BHB = eig_vals_max[..., None, None] * torch.eye(BHB.shape[-1], device=BHB.device, dtype=BHB.dtype) - BHB
        mps, eigen_vals = power_method_matrix(BHB, num_iter=max_iter, verbose=verbose)
        eigen_vals = eig_vals_max - eigen_vals

        # Phase relative to first map
        mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-8))
        if crp:
            mps *= eigen_vals < crp
    else:
        BHB = -BHB
        mps_all = []
        evals_all = []
        for i in range(sets_of_maps):
            
            # Power iterations
            mps, eigen_vals = power_method_matrix(BHB, num_iter=max_iter, verbose=verbose)
            
            # Phase relative to first map and crop
            mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-8))
            if crp:
                mps *= eigen_vals > crp
                
            # Update
            mps_all.append(mps)
            evals_all.append(eigen_vals)
            BHB -= einsum(mps * eigen_vals, mps.conj(), 'Cl ..., Cr ... -> ... Cl Cr')
            
        if sets_of_maps == 1:
            mps = mps_all[0]
            eigen_vals = evals_all[0]
        else:
            mps = torch.stack(mps_all, dim=0)
            eigen_vals = torch.stack(evals_all, dim=0)

    return mps, eigen_vals

def calc_image_covariance_kernels(grappa_kernels: torch.Tensor,
                                  source_vectors: torch.Tensor,
                                  im_size: tuple,
                                  nufft: Optional[NUFFT] = None,
                                  coil_batch: Optional[int] = None,
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
    nufft : NUFFT
        nufft object
    coil_batch : int
        number of coils to process at once
    sigpy_nufft : bool
        use sigpy nufft instead of torchkbnufft
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    BHB : torch.Tensor
        image covariance kernels with shape (*im_size, ncoil, ncoil)
    """

    # Consts
    device = grappa_kernels.device
    nkerns, ncoil, _, num_inputs = grappa_kernels.shape
    assert nkerns == source_vectors.shape[0]
    assert num_inputs == source_vectors.shape[1]
    assert device == source_vectors.device
    if coil_batch is None or coil_batch > ncoil ** 2:
        coil_batch = ncoil

    # Default nufft
    if nufft is None:
        nufft = sigpy_nufft(im_size)
    source_vectors = nufft.rescale_trajectory(source_vectors)

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

    # Build BHB matrix
    BHB = torch.zeros((ncoil * ncoil, *im_size), dtype=grappa_kerns_rs.dtype, device=device)
    kbn = nufft.adjoint
    source_vectors = rearrange(source_vectors, 'N ninp d -> 1 (N ninp) d')
    source_vectors_cross = rearrange(source_vectors_cross, 'N ninp ninp2 d -> 1 (N ninp ninp2) d')
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

def calc_espirit_kernels(ksp_cal: torch.Tensor,
                         im_size: tuple,
                         thresh: Optional[float] = 0.02,
                         kernel_width: Optional[int] = 6,
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

    # Get calibration matrix.
    # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
    mat = array_to_blocks(
        ksp_cal, [kernel_width] * img_ndim, [1] * img_ndim
    )
    calib_mat = mat.reshape((num_coils, -1, *((kernel_width,)*img_ndim))) # For debug
    mat = mat.reshape(num_coils, -1, kernel_width**img_ndim)
    mat = mat.permute(1, 0, 2).reshape(-1, num_coils * kernel_width**img_ndim)

    # Perform SVD on calibration matrix
    if verbose:
        print('Computing SVD on calibration matrix: ', end='')
        start = time.perf_counter()
    _, S, VH = torch.linalg.svd(mat, full_matrices=False)
    VH = VH[S < thresh * S.max(), :]
    if verbose:
        end = time.perf_counter()
        print(f'{end - start:.3f}s')

    # Get kernels
    num_kernels = len(VH)
    kernels = VH.reshape(
        [num_kernels, num_coils] + [kernel_width] * img_ndim)
    
    return kernels, rearrange(calib_mat, 'C N ... -> N C ...')
