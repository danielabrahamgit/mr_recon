# TODO work in progress
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from einops import rearrange, einsum
from mr_recon.utils import (
    rotation_matrix,
    quantize_data,
    gen_grd
)
from mr_recon.fourier import fft, ifft
from mr_recon.imperfections.imperfection import imperfection

def affine_mats_from_parms(motion_params: torch.Tensor) -> torch.Tensor:
    """
    Returns rotation matrices from angles

    Parameters:
    -----------
    motion_params : torch.Tensor
        The motion parameters with shape (..., 6). 
        First three motion params are translations in x, y, z in range [-1, 1].
        Last three motion params are rotations in x, y, z in degrees.
    
    Returns:
    --------
    affine_mats : torch.Tensor
        The affine matrices with shape (..., 3, 4)
        where the first 3 columns is the rotation matrix, 
        and last column is the translation
    """

    # Split into translations and rotations
    batch_size = motion_params.shape[:-1]
    motion_params = motion_params.reshape(-1, 6)
    angles = motion_params[..., 3:]
    shifts = motion_params[..., :3]

    # Affine mat to return
    affine_mats = torch.zeros((motion_params.shape[0], 3, 4), 
                              device=motion_params.device, 
                              dtype=motion_params.dtype)

    # Make rotation matrices
    tup = (None,) * len(batch_size) + (slice(None),)
    ax_x = torch.tensor([1.0, 0, 0], device=motion_params.device, dtype=motion_params.dtype)[tup]
    ax_y = torch.tensor([0, 1.0, 0], device=motion_params.device, dtype=motion_params.dtype)[tup]
    ax_z = torch.tensor([0, 0, 1.0], device=motion_params.device, dtype=motion_params.dtype)[tup]
    R_x = rotation_matrix(axis=ax_x, 
                              theta=torch.deg2rad(angles[..., 0]))
    R_y = rotation_matrix(axis=ax_y, 
                            theta=torch.deg2rad(angles[..., 1]))
    R_z = rotation_matrix(axis=ax_z, 
                            theta=torch.deg2rad(angles[..., 2]))
    R = R_x @ R_y @ R_z

    # Append translations
    affine_mats[..., :3, :3] = R
    affine_mats[..., :3, -1] = shifts

    return affine_mats.reshape((*batch_size, 3, 4))

def apply_affine_3d(imgs: torch.Tensor,
                    affine_mats: torch.Tensor) -> torch.Tensor:
    """
    Applies affine transformations to 3D volumes

    Parameters:
    -----------
    imgs : torch.Tensor
        The volumes with shape (N, *im_batch, nx, ny, nz)
        where nx ny nz are the spatial image dims
    affine_mats : torch.Tensor
        The affine matrices with shape (N, *affine_batch, 3, 4)
        where the first 3 columns is the rotation matrix, 
        and last column is the translation
    
    Returns:
    --------
    tformed_imgs : torch.Tensor
        The transformed volumes with shape (N, *im_batch, *affine_batch, *im_size)
    """

    # Remember original shapes
    im_batch = imgs.shape[1:-3]
    affine_batch = affine_mats.shape[1:-2]
    N = imgs.shape[0]
    assert imgs.shape[0] == affine_mats.shape[0]

    # Flip image from x y z to z y x
    imgs = rearrange(imgs, '... nx ny nz -> ... nz ny nx')
    im_size = imgs.shape[-3:]

    # Flatten
    imgs = imgs.reshape(N, -1, *im_size)
    affine_mats = affine_mats.reshape(N, -1, 3, 4)
    A = affine_mats.shape[1]
    C = imgs.shape[1]

    # Apply affine transformations
    align_corners = False
    grid = F.affine_grid(affine_mats.reshape(-1, 3, 4), 
                         (N * A, imgs.shape[1], *im_size), align_corners=align_corners)
    grid = rearrange(grid, '(N A) nz ny nx three -> N (A nz) ny nx three',
                     N=N, A=A)
    tformed_imgs_real = F.grid_sample(imgs.real, grid, align_corners=align_corners)
    tformed_imgs_imag = F.grid_sample(imgs.imag, grid, align_corners=align_corners)
    tformed_imgs = tformed_imgs_real + 1j * tformed_imgs_imag

    # Reshape batch dims
    tformed_imgs = rearrange(tformed_imgs, 'N C (A nz) ny nx -> A C nz ny nx N',
                             A=A, nz=im_size[0])
    tformed_imgs = tformed_imgs.reshape((*affine_batch, *tformed_imgs.shape[1:]))
    tformed_imgs = rearrange(tformed_imgs, '... C nz ny nx N -> C ... nz ny nx N')
    tformed_imgs = tformed_imgs.reshape((*im_batch, *tformed_imgs.shape[1:]))
    tformed_imgs = rearrange(tformed_imgs, '... nz ny nx N -> N ... nz ny nx')

    # Flip back to x y z
    tformed_imgs = rearrange(tformed_imgs, '... nz ny nx -> ... nx ny nz')

    return tformed_imgs

class motion_imperfection(imperfection):

    def __init__(self,
                 motion_params: torch.Tensor,
                 mps_unmasked: torch.Tensor,
                 mask: torch.Tensor,
                 L: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
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
            First three motion params are translations in x, y, z in range [-1, 1]. 
            Last three motion params are rotations in x, y, z in degrees.
        mps_unmasked : torch.Tensor
            The coil sensitivity maps with shape (nc, *im_size)
            Make sure these are unmasked!
        mask : torch.Tensor
            The mask with shape (*im_size)
        L : int 
            The rank of the low-rank model.
        method : str
            'ts' - time segmentation
            'svd' - SVD based splitting
        interp_type : str
            'zero' - zero order interpolator
            'linear' - linear interpolator 
            'lstsq' - least squares interpolator
        verbose : bool
            toggles print statements
        """
        assert mps_unmasked.device == motion_params.device
        assert mps_unmasked.device == mask.device

        self.motion_params = motion_params
        self.mps = mps_unmasked
        self.mask = mask
        self.torch_dev = mps_unmasked.device
        self.trj_size = motion_params.shape[:-1]
        self.im_size = mps_unmasked.shape[1:]
        assert len(self.im_size) == 2 or len(self.im_size) == 3
        if len(self.im_size) == 2:
            self.im_size = (*self.im_size, 1)
            self.is_2d = True
        else:
            self.is_2d = False
          
        super().__init__(L, method, interp_type, verbose)
    
    def _calc_time_segmentation(self) -> None:
        """
        Computes both h_l and T_l functions using time segmentation.

        T_l{x(r)} = Motion_theta_l { x(r) }
        h_l(t) = self.interp_funcs(l, t)

        Saves:
        ------
        self.motion_clusters : torch.Tensor
            The clusters of motion states with shape (L, 6)
        self.affine_mats : torch.Tensor
            The affine transformation matrices with shape (L, 3, 4)
        self.temporal_funcs : torch.Tensor
            The interpolating functions with shape (L, *trj_size)
        """

        # Cluster motion states
        self.motion_clusters, idxs = quantize_data(self.motion_params.reshape((-1, 6)), 
                                                   self.L, method='cluster')
        idxs = idxs.reshape(self.trj_size)
        self.affine_mats = affine_mats_from_parms(self.motion_clusters)
        self.affine_mats_adj = self.affine_mats.clone()
        self.affine_mats_adj[..., :3] = self.affine_mats_adj[..., :3].mT
        # TODO which one?
        self.affine_mats_adj[..., -1:] *= -1
        self.affine_mats_adj[..., -1:] = self.affine_mats_adj[..., :3] @ self.affine_mats_adj[..., -1:]
        
        if 'lstsq' in self.interp_type:
            raise NotImplementedError("Least Squares interpolation not implemented yet")
        elif 'linear' in self.interp_type:
            raise NotImplementedError("Linear interpolation not implemented yet")
        elif 'zero' in self.interp_type:
            # Indicator function
            interp_funcs = torch.zeros((self.L, *self.trj_size,), dtype=torch.complex64, device=self.torch_dev)
            for i in range(self.L):
                interp_funcs[i, ...] = 1.0 * (idxs == i)

        self.temporal_funcs = interp_funcs
    
    # TODO
    def apply_spatio_temporal(self,
                              x: torch.Tensor,
                              r_inds: Optional[torch.Tensor] = slice(None), 
                              t_inds: Optional[torch.Tensor] = slice(None),
                              lowrank: Optional[bool] = False) -> torch.Tensor:
        """
        Applies the full spatio-temporal imperfection model to the data

        Parameters:
        -----------
        x : torch.Tensor
            The input data with shape (..., nc, *im_size)
        r_inds : Optional[tuple]
            Slices the flattened spatial terms with shape (N,)
        t_inds : Optional[tuple]
            Slices the flattened temporal terms with shape (N,)
        lowrank : Optional[bool]
            if True, uses lowrank approximation
        
        Returns:
        --------
        xt : torch.Tensor
            The spatio-temporal data with shape (..., nc, N)
        """
        
        # Flatten spatial and temporal dims
        x = x.flatten(start_dim=-len(self.im_size))[..., r_inds]

        if lowrank:
            h = self.apply_temporal_adjoint().reshape((self.L, -1))[:, t_inds]
            b = self.apply_spatial().reshape((self.L, -1))[:, r_inds]
            xt = torch.sum(b * h, dim=0) * x
        else:
            alphas = self.alphas.reshape((self.B, -1))[:, t_inds]
            phis = self.phis.reshape((self.B, -1))[:, r_inds]
            xt = torch.exp(-2j * torch.pi * torch.sum(alphas * phis, dim=0)) * x
        return xt
    
    def apply_spatial(self, 
                      x: Optional[torch.Tensor] = None, 
                      ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        if self.method == 'ts':
            if self.is_2d:
                x = x[..., None]
            y = apply_affine_3d(x[None, ...], self.affine_mats[ls][None,])[0]
            if self.is_2d:
                y = y[..., 0]
        # x -> ... nc *im_size
        return y # -> ... nc len(ls) *im_size
    
    def apply_spatial_adjoint(self, 
                              y: torch.Tensor, 
                              ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        # y -> ... nc len(ls) *im_size
        if self.method == 'ts':
            if self.is_2d:
                y = y[..., None]
            y_rs = y.moveaxis(-len(self.im_size)-1, 0) # -> len(ls) ... nc *im_size
            x = apply_affine_3d(y_rs, self.affine_mats_adj[ls][:, None, ...]) # -> len(ls) 1 ... nc *im_size
            x = einsum(x, 'L one ... -> one ...')[0]
            if self.is_2d:
                x = x[..., 0]
        return x
        
    def apply_temporal(self, 
                       x: torch.Tensor, 
                       ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        # x -> ... nc L *trj_size
        h = self.temporal_funcs[ls]
        return (h * x).sum(dim=-len(self.trj_size)-1)

    def apply_temporal_adjoint(self, 
                               y: Optional[torch.Tensor] = None, 
                               ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        h = self.temporal_funcs[ls]
        return h.conj() * y.unsqueeze(-len(self.trj_size)-1)


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

class motion_op(nn.Module):
    
    def __init__(self,
                 im_size: tuple,
                 fovs: Optional[tuple] = None,
                 reuse_UV: Optional[bool] = False):
        """
        Parameters:
        -----------
        im_size : tuple
            The image dimensions
        fovs : tuple
            The field of views, same size as im_size
        reuse_UV : bool
            Whether to reuse U and V operators after consecutive calls
        """
        super(motion_op, self).__init__()

        # Consts
        d = len(im_size)
        assert d == 2 or d == 3
        if fovs is None:
            fovs = (1,) * d
        assert d == len(fovs)
    
        # Gen grid
        fovsk = tuple([im_size[i] / fovs[i] for i in range(d)])
        rgrid = gen_grd(im_size, fovs=fovs) # ... d from (-FOV/2, FOV/2)
        kgrid = 2 * torch.pi * gen_grd(im_size, fovsk) # ... d from (-N/2, N/2) / FOV
        if d == 2:
            rgrid = torch.cat((rgrid, rgrid[..., :1] * 0), dim=-1) # ... 3
            kgrid = torch.cat((kgrid, kgrid[..., :1] * 0), dim=-1) # ... 3

        # Save
        self.im_size = im_size
        self.kgrid = nn.Parameter(kgrid.type(torch.float32), requires_grad=False)
        self.rgrid = nn.Parameter(rgrid.type(torch.float32), requires_grad=False)
        # self.rgrid = rgrid.type(torch.float32)
        # self.kgrid = kgrid.type(torch.float32)
        self.U = None
        self.V = None
        self.reuse_UV = reuse_UV
    
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
        if self.reuse_UV:
            self.U = U
            self.V = V

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