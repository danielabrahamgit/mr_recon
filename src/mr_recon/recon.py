import os
import torch
import time
import numpy as np
import sigpy as sp

from tqdm import tqdm
from typing import Optional
from mr_recon.algs import (
    density_compensation, 
    conjugate_gradient, 
    largest_eigenvalue, 
    FISTA
)


""""
In the comments and code, we use a few acronyms/shortened nouns. 
Here is a list and description of each:
    nx   - number of pixels in the x-direction
    ny   - number of pixels in the y-direction
    nz   - number of pixels in the z-direction
    nc   - number of coil sensitivity maps
    nro  - number of points along the readout dimenstion
    npe  - number of phase encodes/groups/interleaves
    ntr  - number of TRs or time points for temporal subspace recons
    nsub - number of subspace coefficients
    d    - dimension of the problem. d = 2 for 2D, d = 3 for 3D, etc
"""

class recon:
    """
    Framework for running, benchmarking, and plotting reconstructions.
    """

    def __init__(self, 
                 device_idx: Optional[int] = None,
                 verbose: Optional[bool] = True):
        """
        Parameters
        ----------
        trj : np.ndarray <float> 
            k-space trajectory with shape (nro, npe, ntr, d)
            axis i is scaled between +-image_size[i]/2
        mps : np.ndarray <complex64>
            sensitivity maps with shape (nc, *im_size)
        dcf : np.ndarray <float>
            densitity compensation function with shape (nro, npe, ntr)
        phi : np.ndarray <complex>
            subspace basis with shape (nsub, ntr)
        device_idx : int
            gpu device index, defualt is CPU
        verbose : bool
            Toggles print statements
        """

        # GPU/CPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        self.device_idx = device_idx
        self.device = sp.Device(device_idx)
        self.mvd = lambda x : sp.to_device(x, self.device)
        self.mvc = lambda x : sp.to_device(x, sp.cpu_device)
        self.xp = self.device.xp
        if device_idx is None or device_idx == -1:
            self.torch_dev = torch.device('cpu')
        else:
            self.torch_dev = torch.device(device_idx)
        self.verbose = verbose

    def run_recon(self,
                  A_linop: callable,
                  ksp: np.ndarray,
                  proxg: Optional[callable] = None,
                  max_eigen: Optional[float] = None,
                  max_iter: Optional[int] = 15,
                  lamda_l2: Optional[float] = 0.0) -> np.ndarray:
        """
        Runs reconstruction using provided linear operator and data

        Parameters
        ----------
        A_linop : callable
            The A operator with functions A.forward, A.adjoint, and A.normal
        ksp : np.ndarray <complex>
            k-space raw data with shape (nc, nro, npe, ntr)
        proxg : callable
            proximal operator. If given, we will do FISTA, otherwise, we do CG
        max_eigen : float
            maximum eigenvalue of AHA
        max_iter : int
            max number of iterations for recon algorithm
        lamda_l2 : float
            l2 lamda regularization for SENSE: ||Ax - b||_2^2 + lamda_l2||x||_2^2

        Returns
        ---------
        recon : np.ndarray <complex>
            the reconstructed image/volume/subspace coefficients.
            has shape (nsub, nx, ny, (nz))
        """

        # Estimate largest eigenvalue so that lambda max of AHA is 1
        if max_eigen is None:
            x0 = torch.randn(A_linop.ishape, dtype=torch.complex64, device=self.torch_dev)
            max_eigen = largest_eigenvalue(A_linop.normal, x0, verbose=self.verbose)
            max_eigen *= 1.01
        else:
            max_eigen = 1.0

        # Starting point
        start = time.perf_counter()
        y = torch.tensor(ksp, dtype=torch.complex64, device=self.torch_dev)
        y = y / torch.norm(y)
        AHb = A_linop.adjoint(y) / (max_eigen ** 0.5)
        end = time.perf_counter()
        if self.verbose:
            print(f'AHb took {end-start:.3f}(s)')

        # Wrap normal with max eigen
        AHA = lambda x : A_linop.normal(x) / max_eigen

        # FISTA
        if proxg:
            coeffs = FISTA(AHA=AHA, 
                           AHb=AHb, 
                           proxg=proxg, 
                           num_iters=max_iter,
                           verbose=self.verbose)
        # CG
        else:
            coeffs = conjugate_gradient(AHA=AHA, 
                                        AHb=AHb,
                                        num_iters=max_iter,
                                        lamda_l2=lamda_l2,
                                        verbose=self.verbose)

        return coeffs.cpu().numpy()

    def run_mfi_recon(self,
                      A_linop: callable,
                      ksp: np.ndarray,
                      b0_dct: dict,
                      proxg: Optional[callable] = None,
                      max_eigen: Optional[float] = None,
                      max_iter: Optional[int] = 15,
                      lamda_l2: Optional[float] = 0.0) -> np.ndarray:
        """
        Runs reconstruction using provided linear operator and data

        Parameters
        ----------
        A_linop : callable
            The A operator with functions A.forward, A.adjoint, and A.normal
            [optional] has an A.max_eigen variable
        ksp : np.ndarray <complex>
            k-space raw data with shape (nc, nro, npe, ntr)
        b0_dct: dictionary
            b0_map: np.ndarray <float>
                The b0 map (hertz) with shape (nx, ny, (nz))
            dt: float
                The dwell time (seconds) between samples
            nseg: int
                Number of MFI segments
        proxg : callable
            proximal operator. If given, we will do FISTA, otherwise, we do CG
        max_eigen : float
            maximum eigenvalue of AHA
        max_iter : int
            max number of iterations for recon algorithm
        lamda_l2 : float
            l2 lamda regularization for SENSE: ||Ax - b||_2^2 + lamda_l2||x||_2^2

        Returns
        ---------
        recon : np.ndarray <complex>
            the reconstructed image/volume/subspace coefficients.
            has shape (nsub, nx, ny, (nz))
        """

        # Extract from b0 dict
        b0_map = -b0_dct['b0_map'].copy()
        dt = b0_dct['dt']
        nseg = 2 * (b0_dct['nseg']//2) + 1

        # Make desired freqs
        df = 0.1
        t = torch.arange(ksp.shape[1]) * dt
        desired_freqs = torch.arange(b0_map.min(), b0_map.max(), df)
        D = torch.exp(-2j * torch.pi * desired_freqs[None, :] * t[:, None]) / (len(t))

        use_svd = False
        if use_svd:
            # Decompose exp phase
            from tensorly.tenalg import svd_interface
            U, s, Vt = svd_interface(D, n_eigenvecs=nseg)
            B = torch.from_numpy(U)
            coefs = torch.from_numpy(s[:, None] * Vt)
        else:
            # Make basis freqs
            mx_freq = int(np.abs(b0_map).max())
            base_freqs = torch.linspace(-mx_freq, mx_freq, nseg)
            B = torch.exp(-2j * torch.pi * base_freqs[None, :] * t[:, None]) / (len(t))

            # Least squares fit TODO GPU?
            coefs = torch.from_numpy(np.linalg.lstsq(B.H @ B, B.H @ D, rcond=None)[0])
            # coefs = torch.linalg.lstsq(B.H @ B, B.H @ D).solution 

        # Quantize b0
        tup = (None,) * b0_map.ndim + (slice(None),)
        diffs = torch.abs(torch.from_numpy(b0_map)[..., None] \
                          - desired_freqs[tup])
        b0_map_inds = torch.argmin(diffs, dim=-1)

        # MFI Recons
        mfi_imgs = None
        self.verbose = False
        for k in tqdm(range(nseg), 'MFI Recons'):

            # Recon 
            ksp_mfi = ksp * B[None, :, k, None, None].numpy()
            img = self.run_recon(A_linop=A_linop,
                                 ksp=ksp_mfi,
                                 proxg=proxg,
                                 max_eigen=max_eigen,
                                 max_iter=max_iter,
                                 lamda_l2=lamda_l2)
            img = torch.from_numpy(img)
            if mfi_imgs is None:
                mfi_imgs = torch.zeros((nseg, *img.shape), dtype=img.dtype)
            mfi_imgs[k] = img
        
        # MFI interpolate
        img_final = torch.zeros_like(img)
        if img.ndim != b0_map_inds.ndim:
            b0_map_inds = np.repeat(b0_map_inds.numpy()[None, ...], img.shape[0], axis=0)
            b0_map_inds = torch.from_numpy(b0_map_inds)
        for k in tqdm(range(0, len(desired_freqs)), 'MFI Combine'):
            
            # Select mask for this frequency
            msk = 1.0 * (b0_map_inds == k)

            # Compute weighted images 
            tup = (slice(None),) + (None,) * img.ndim + (k,)
            img_weighted = mfi_imgs * coefs[tup] * msk[None, ...]
            
            # Append to final image
            img_final += torch.sum(img_weighted, dim=0)

        return img_final.numpy()
