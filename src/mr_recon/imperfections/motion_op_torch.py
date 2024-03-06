import torch

from typing import Optional
from einops import einsum
from torch import fft as fft_torch

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

def fft(x, dim=None, norm='ortho'):
    """Matches Sigpy's fft, but in torch"""
    x = fft_torch.ifftshift(x, dim=dim)
    x = fft_torch.fftn(x, dim=dim, norm=norm)
    x = fft_torch.fftshift(x, dim=dim)
    return x

def ifft(x, dim=None, norm='ortho'):
    """Matches Sigpy's fft adjoint, but in torch"""
    x = fft_torch.ifftshift(x, dim=dim)
    x = fft_torch.ifftn(x, dim=dim, norm=norm)
    x = fft_torch.fftshift(x, dim=dim)
    return x

class motion_op(torch.nn.Module):
    
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
