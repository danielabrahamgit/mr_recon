# TODO work in progress

import torch
import sigpy as sp

from tqdm import tqdm
from typing import Optional, Union
from einops import rearrange, einsum
from mr_recon.imperfections.motion_op_torch import rigid_motion
from mr_recon.utils import (
    batch_iterator, 
    torch_to_np, 
    np_to_torch,
    apply_window,
    rotation_matrix,
    quantize_data
)
from mr_recon.fourier import fft, ifft
from mr_recon.algs import lin_solve

class motion_handler:

    def __init__(self,
                 motion_params: torch.Tensor,
                 mps: torch.Tensor,
                 nseg: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 quant_type: Optional[str] = 'uniform',
                 verbose: Optional[bool] = True):
        """
        Motion handler class for rigid motion correction in MR images. 

        Our motion model takes the form 
        moved_vec = Rotate(vec) + translation

        where Rotate(vec) = Rotate_x(Rotate_y(Rotate_z(vec)))

        Parameters:
        -----------
        motion_params : torch.Tensor
            The motion parameters with shape (*trj_size, 6). 
            First three motion params are translations in x, y, z in mm. 
            Last three motion params are rotations in x, y, z in degrees.
        mps : torch.Tensor
            The coil sensitivity maps with shape (nc, *im_size)
        nseg : int
            number of segments
        method : str
            'ts' - time segmentation
            'svd' - SVD based splitting
        interp_type : str
            'zero' - zero order interpolator
            'linear' - linear interpolator 
            'lstsq' - least squares interpolator
        quant_type : str
            'cluster' - uses k-means to optimally find centers
            'uniform' - uniformly spaced bins
        verbose : bool
            toggles print statements
        """
        msg = 'mps and motion params must be on the same device'
        assert mps.device == motion_params.device, msg

        self.motion_params = motion_params
        self.mps = mps
        self.nseg = nseg
        self.torch_dev = mps.device
        self.method = method.lower()
        self.interp_type = interp_type.lower()
        self.trj_size = motion_params.shape[:-1]
        self.im_size = mps.shape[1:]
        self.verbose = verbose
        self.spatial_batch_size = 2 ** 8

        if 'ts' in method.lower():
            # Compute time segmentation
            self.motion_clusters = self._quantize_data(motion_params.reshape((-1, self.nbasis)), 
                                                       nseg, method=quant_type)
            self.interp_coeffs = self._calc_motion_interpolators(self.motion_clusters, interp_type)
        else:
            raise NotImplementedError
    
    def correct_trj_ksp(self,
                        trj: torch.Tensor,
                        ksp: torch.Tensor):
        """
        Correct the k-space data for translation and trajectory for rotation.

        Note that this is not a full correction, since we still need 
        to apply the rotation operator to the coils in the recon.

        Parameters:
        -----------
        trj : torch.Tensor
            The trajectory data with shape (*trj_size, 3)
        ksp : torch.Tensor
            The k-space data with shape (nc, *trj_size)

        Returns:
        --------
        trj_rot : torch.Tensor
            The corrected trajectory data with shape (*trj_size, 3)
        ksp_rot : torch.Tensor
            The corrected k-space data with shape (nc, *trj_size)
        """

        # Make sure motion parameters are broadcastable to trj size
        motion_params = self.motion_params
        trj_size = trj.shape[:-1]
        assert ksp.shape[1:] == trj_size, "Trajectory and k-space data must have same shape"
        assert is_broadcastable(motion_params.shape, trj_size), "Motion parameters must have broadcastable shape to trajectory"

        # Correct for rotation
        rots = motion_params[..., 3:]
        tup = (None) * (len(trj_size)) + (slice(None),)
        ax_x = torch.tensor([1.0, 0, 0])[tup]
        ax_y = torch.tensor([0, 1.0, 0])[tup]
        ax_z = torch.tensor([0, 0, 1.0])[tup]
        R_x = rotation_matrix(axis=ax_x, 
                              theta=torch.deg2rad(rots[..., 0]))
        R_y = rotation_matrix(axis=ax_y, 
                              theta=torch.deg2rad(rots[..., 1]))
        R_z = rotation_matrix(axis=ax_z, 
                              theta=torch.deg2rad(rots[..., 2]))
        R = R_x @ R_y @ R_z
        trj_rot = einsum(R, trj, '... 3out 3in, ... 3in -> ... 3out')

        # Correct k-space data for translation
        shifts = motion_params[..., :3]
        dot = einsum(trj_rot, shifts, '... 3out, ... 3out -> ...')
        ksp_rot = ksp * torch.exp(-2j * torch.pi * dot)

        return trj_rot, ksp_rot
    
    def _gen_grid(self) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates grids needed for 3 shear model

        Returns:
        --------
        rGrids : torch.Tensor
            The real space grid
        kGrids : torch.Tensor
            The k-space grid
        rkGrids : torch.Tensor
            The hybrid-space grid
        """
        rGrids = []
        kGrids = []
        rkGrids = []
        for ax_ in range(len(ishape)):
            min_k = -np.floor(ishape[ax_] / 2) 
            max_k = np.ceil(ishape[ax_] / 2) 
            Nk = max_k - min_k
            
            r_m = torch.arange(min_k, max_k, 1)
            if len(r_m)== 0:
                r_m = torch.Tensor([0.])
            kGrids.append(2 / Nk * np.pi * r_m)
            rGrids.append(r_m)

        per =  [ [0, 2, 1], [1, 0, 2] ]
        for ii in range(2):
            rk = []
            for jj in range(3):
                rk.append( torch.outer(kGrids[per[ii][jj]], rGrids[per[1-ii][jj]]) )
            rkGrids.append(rk)
            
        return rGrids, kGrids, rkGrids
 
    # def _calc_motion_interpolators(self,
    #                                motion_clusters: torch.Tensor,
    #                                interp_type: Optional[str] = 'zero') -> torch.Tensor:
    #     """
    #     Calculates motion interpolation coefficients h_l(motion_params) 
    #     from the following model:
        
    #     mps(Rotate_theta(r - t)) = sum_l h_l(theta, t) * mps(Rotate_theta_l(r - t_l))

    #     Parameters:
    #     -----------
    #     motion_clusters : torch.tensor <float32>
    #         motion segmentation coeffs with shape (nseg, 6)
    #     interp_type : str
    #         interpolator type from the list
    #             'zero' - zero order interpolator
    #             'linear' - linear interpolator 
    #             'lstsq' - least squares interpolator
        
    #     Returns:
    #     --------
    #     interp_coeffs : torch.tensor <complex64>
    #         interpolation coefficients 'h_l(t)' with shape (*trj_size, nseg)
    #     """

    #     # Consts
    #     assert self.torch_dev == motion_clusters.device
    #     nseg = motion_clusters.shape[0]
    #     trj_size = self.trj_size

    #     # Least squares interpolator
    #     if 'lstsq' in interp_type:

    #         # Prep AHA, AHB matrices
    #         motion_flt = rearrange(self.motion_params, '... 6 -> (...) 6').type(torch.float32)
    #         mps_flt = rearrange(self.mps, 'nc ... -> nc (...)').type(torch.complex64)
    #         AHA = torch.zeros((nseg, nseg), 
    #                           dtype=torch.complex64, device=self.torch_dev)
    #         AHb = torch.zeros((nseg, motion_flt.shape[0]), 
    #                           dtype=torch.complex64, device=self.torch_dev)
            
    #         # Compute AHA and AHB in batches
    #         N = mps_flt.shape[1]
    #         batch_size = self.spatial_batch_size
    #         for n1 in tqdm(range(0, N, batch_size), 'Least Squares Interpolators'):
    #             n2 = min(n1 + batch_size, N)

    #             # Accumulate AHA
                
    #             AHA += A_batch.H @ A_batch / (n2 - n1)

    #             # Accumulate AHb
                
    #             AHb += A_batch.H @ B_batch / (n2 - n1)

    #         # Solve for x = (AHA)^{-1} AHb
    #         x = lin_solve(AHA, AHb, solver='pinv')
            
    #         # Reshape (nseg, T)
    #         interp_coeffs = x.T.reshape((*trj_size, nseg))
            
    #     # Linear interpolator
    #     elif 'lin' in interp_type:
    #         raise NotImplementedError
        
    #     # Zero order hold/nearest interpolator
    #     else:
        
    #         # Empty return coefficients
    #         interp_coeffs = torch.zeros((*trj_size, nseg), dtype=torch.float32, device=self.torch_dev)

    #         # Find closest points
    #         tup = (slice(None),) + (None,) * (self.alphas.ndim - 1) + (slice(None),)
    #         inds = torch.argmin(
    #             torch.linalg.norm(self.alphas[None, ...] - betas[tup], dim=-1),
    #             dim=0) # *trj_size -> values in [0, ..., nseg-1]
            
    #         # Indicator function
    #         for i in range(nseg):
    #             interp_coeffs[..., i] = 1.0 * (inds == i)
            
    #         del inds

    #     return interp_coeffs.type(torch.complex64)
  
   