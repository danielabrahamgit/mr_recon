# TODO work in progress

import torch
import sigpy as sp

from tqdm import tqdm
from typing import Optional, Union
from einops import rearrange, einsum
from mr_recon.utils import (
    batch_iterator, 
    torch_to_np, 
    np_to_torch,
    apply_window,
    rotation_matrix,
    quantize_data,
    gen_grd
)
from mr_recon.fourier import fft, ifft

class motion_handler:

    def __init__(self,
                 motion_params: torch.Tensor,
                 mps_unmasked: torch.Tensor,
                 mask: torch.Tensor,
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
            First three motion params are translations in x, y, z in meters. 
            Last three motion params are rotations in x, y, z in degrees.
        mps_unmasked : torch.Tensor
            The coil sensitivity maps with shape (nc, *im_size)
            Make sure these are unmasked!
        mask : torch.Tensor
            The mask with shape (*im_size)
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
        assert mps_unmasked.device == motion_params.device
        assert mps_unmasked.device == mask.device

        self.motion_params = motion_params
        self.mps = mps_unmasked
        self.mask = mask
        self.nseg = nseg
        self.torch_dev = mps_unmasked.device
        self.method = method.lower()
        self.interp_type = interp_type.lower()
        self.trj_size = motion_params.shape[:-1]
        self.im_size = mps_unmasked.shape[1:]
        self.verbose = verbose
        self.spatial_batch_size = 2 ** 8

        # Cluster motion states
        self.motion_clusters = quantize_data(motion_params.reshape((-1, self.nbasis)), 
                                                nseg, method=quant_type)
        
        if 'ts' in method.lower():
            # Compute time segmentation
            self.interp_coeffs = self._calc_motion_interpolators(self.motion_clusters, interp_type)
        elif 'svd' in method.lower():
            # Compute SVD based splitting
            self.
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

class motion_op(torch.nn.Module):
    
    def __init__(self,
                 im_size: tuple,
                 fovs: Optional[tuple] = None):
        """
        Parameters:
        -----------
        im_size : tuple
            The image dimensions
        fovs : tuple
            The field of views, same size as im_size
        """
        super(motion_op, self).__init__()

        # Consts
        d = len(im_size)
        assert d == len(fovs)
        assert d == 2 or d == 3
        if fovs is None:
            fovs = (1,) * d
    
        # Gen grid
        fovsk = tuple([im_size[i] / fovs[i] for i in range(d)])
        rgrid = gen_grd(im_size, fovs=fovs) # ... d from (-FOV/2, FOV/2)
        kgrid = 2 * torch.pi * gen_grd(im_size, fovsk) # ... d from (-N/2, N/2) / FOV
        if d == 2:
            rgrid = torch.cat((rgrid, rgrid[..., :1] * 0), dim=-1) # ... 3
            kgrid = torch.cat((kgrid, kgrid[..., :1] * 0), dim=-1) # ... 3
        # Save
        self.im_size = im_size
        self.rgrid = rgrid.type(torch.float32)
        self.kgrid = kgrid.type(torch.float32)
        self.U = None
        self.V = None
    
    def _build_U(self,
                 translations: torch.Tensor) -> torch.Tensor:
        """
        Builds U terms from translation parameters

        Parameters:
        -----------
        translations : torch.Tensor
            The translations with shape (N, 3), x y z in meters

        Returns:
        --------
        U : torch.Tensor
            The U terms with shape (N, *im_size)
        """

        return torch.exp(-1j * einsum(self.kgrid, translations.type(torch.float32), 
                                      '... out, N out -> N ...'))
    
    def _build_V(self,
                 rotations: torch.Tensor) -> torch.Tensor:
        """
        Builds V terms from rotation parameters

        Parameters:
        -----------
        rotations : torch.Tensor
            The rotations with shape (N, 3), x y z in degrees

        Returns:
        --------
        V : torch.Tensor
            The V terms with shape (6, N, *im_size)
        """

        roations_radians = torch.deg2rad(rotations).type(torch.float32)
        v_1_tan = torch.tan(roations_radians[:, 0] / 2) * \
                  self.kgrid[None, ..., 1] * self.rgrid[None, ..., 2]
        v_1_sin = torch.sin(roations_radians[:, 0]) * \
                  -self.kgrid[None, ..., 2] * self.rgrid[None, ..., 1]
        v_2_tan = torch.tan(roations_radians[:, 1] / 2) * \
                  self.kgrid[None, ..., 2] * self.rgrid[None, ..., 0]
        v_2_sin = torch.sin(roations_radians[:, 1]) * \
                  -self.kgrid[None, ..., 0] * self.rgrid[None, ..., 2]
        v_3_tan = torch.tan(roations_radians[:, 2] / 2) * \
                  self.kgrid[None, ..., 0] * self.rgrid[None, ..., 1]
        v_3_sin = torch.sin(roations_radians[:, 2]) * \
                  -self.kgrid[None, ..., 1] * self.rgrid[None, ..., 0]
        V = torch.stack((v_1_tan, v_1_sin,
                         v_2_tan, v_2_sin,
                         v_3_tan, v_3_sin), dim=0)
        V = torch.exp(1j * V)
        return V

    def forward(self,
                img: torch.Tensor,
                motion_params: torch.Tensor) -> torch.Tensor:
        """
        Applies motion operation to the image
        
        Parameters:
        -----------
        img : torch.tensor
            The image with shape (..., *im_size)
        motion_params : torch.Tensor
            The motion parameters with shape (N, 6). 
            First three motion params are translations in x, y, z in meters. 
            Last three motion params are rotations in x, y, z in degrees.
        """

        # Consts
        d = len(self.im_size)
        assert img.shape[-d:] == self.im_size

        # Build U and V operators
        if self.U is None or self.V is None:
            U = self._build_U(motion_params[:, :3]) # (N *im_size)
            V = self._build_V(motion_params[:, 3:]) # (6, N, *im_size)
        else:
            U = self.U
            V = self.V

        # Add empty dims
        img_nbatch_dims = len(img.shape) - d
        tup = (slice(None),) * img_nbatch_dims + (None,) + (slice(None),) * d
        img = img[tup] # (... 1 *im_size)
        if d == 2:
            img = img[..., None]
            V = V[..., None]
            U = U[..., None]
            
        # Apply V
        V_dims = [
            4, 5, 4,
            2, 3, 2,
            0, 1, 0
        ]
        ft_dims = [
            -3, -2, -3,
            -1, -3, -1,
            -2, -1, -2
        ]
        for i in range(9):
            img = ifft(V[V_dims[i]] * fft(img, dim=ft_dims[i]), dim=ft_dims[i])

        # Apply U
        dims = tuple(range(-3, 0))
        img = ifft(U * fft(img, dim=dims), dim=dims)
        
        # Remove empty dim 
        if d == 2:
            img = img[..., 0]

        return img
