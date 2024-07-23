import torch

from typing import Optional
from mr_recon.utils import gen_grd
from mr_recon.imperfections.exponential import exponential_imperfection

def bases(x, y, z):
    assert x.shape == y.shape
    assert z.shape == x.shape
    tup = (None,) + (slice(None),) * x.ndim
    x = x[tup]
    y = y[tup]
    z = z[tup]
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    x3 = x ** 3
    y3 = y ** 3
    z3 = z ** 3
    return torch.cat([
        torch.ones_like(x),
        x,
        y,
        z,
        x * y,
        z * y,
        3 * z2 - (x2 + y2 + z2),
        x * z,
        x2 - y2,
        3 * y * x2 - y3, 
        x * y * z,
        (5 * z2 - (x2 + y2 + z2)) * y,
        5 * z3 - 3 * z * (x2 + y2 + z2),
        (5 * z2 - (x2 + y2 + z2)) * x,
        z * x2 - z * y2,
        x3 - 3 * x * y2
    ], dim=0)
   
class eddy_imperfection(exponential_imperfection):

    def __init__(self, 
                 sph_coeffs: torch.Tensor,
                 im_size: tuple,
                 fov: tuple,
                 skope_inds: torch.Tensor,
                 z_ofs: float,
                 rotations: tuple,
                 L: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 verbose: Optional[bool] = True,
                 coord_bases: Optional[torch.Tensor] = None):
        """
        Parameters:
        -----------
        sph_coeffs : torch.tensor <float>
            spherical harmonic coeffs with shape (*trj_size, nbasis)
            each term corresponds to a basis function (see bases above)
        im_size : tuple
            num voxels in each dim
        fov : tuple
            FOV in meters in each dim
        skope_inds : torch.tensor <int>
            indices of the basis functions to use
        z_ofs : float
            z offset in meters
            only makes sense for 2D
        rotations : tuple
            rotation of imaging slice along x and y axis in degrees
            only makes sense for 2D
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

        # Consts
        d = len(im_size)
        nbasis = sph_coeffs.shape[-1]
        alphas = (sph_coeffs / (2 * torch.pi)).moveaxis(-1, 0)
        if skope_inds is None:
            skope_inds = torch.arange(nbasis)
        assert nbasis == len(skope_inds)
        assert d == 2 or d == 3
        assert d == len(fov)

        # Gen X Y Z grids
        if coord_bases is None:
            grd = gen_grd(im_size, fov).to(alphas.device).type(alphas.dtype)
            if d == 2:
                grd = torch.concatenate((grd, grd[..., :1] * 0), dim=-1)
                
                if rotations is not None:
                    rotations = torch.tensor(rotations, dtype=torch.float32, device=alphas.device)
                    thetas = torch.deg2rad(rotations)
                    Rx = torch.tensor([
                        [1, 0, 0],
                        [0, torch.cos(thetas[0]), -torch.sin(thetas[0])],
                        [0, torch.sin(thetas[0]), torch.cos(thetas[0])]], 
                        device=alphas.device)
                    Ry = torch.tensor([
                        [torch.cos(thetas[1]), 0, torch.sin(thetas[1])],
                        [0, 1, 0],
                        [-torch.sin(thetas[1]), 0, torch.cos(thetas[1])]], 
                        device=alphas.device)
                    grd = (Ry @ Rx @ grd[..., None])[..., 0]

                if z_ofs is not None:
                    grd[..., 2] += z_ofs
            
            # Build phis
            X = grd[..., 0]
            Y = grd[..., 1]
            Z = grd[..., 2]
        else:
            X = coord_bases[..., 0]
            Y = coord_bases[..., 1]
            Z = coord_bases[..., 2]
        skope_inds = skope_inds.to(alphas.device)
        all_bases = bases(X, Y, Z).to(alphas.device)
        phis = all_bases[skope_inds]

        super(eddy_imperfection, self).__init__(phis, alphas, L, method, interp_type, verbose)

    def _calc_svd(self):
        """
        Computed SVD of spatio-temporal operator to compute 
        the spatial and temporal basis functions.

        Returns:
        --------
        spatial_funcs : torch.tensor
            spatial basis functions with shape (nseg, *im_size)
        temporal_funcs : torch.tensor
            temporal basis functions with shape (*trj_size, nseg)
        """
        
        return self._calc_svd_batched()