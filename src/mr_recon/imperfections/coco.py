import torch

from typing import Optional
from mr_recon.utils import gen_grd
from mr_recon.imperfections.exponential import exponential_imperfection

class coco_imperfection(exponential_imperfection):

    def __init__(self, 
                 trj: torch.Tensor,
                 im_size: tuple,
                 fov: tuple,
                 dt: float,
                 B0: float,
                 z_ofs: float,
                 rotations: tuple,
                 L: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 verbose: Optional[bool] = True):
        """
        Parameters:
        -----------
        trj : torch.tensor <float>
            The k-space trajectory with shape (*trj_size, d). 
            we assume that trj values are in [-n/2, n/2] (for nxn grid)
            trj[0] is readout dimension
        im_size : tuple
            num voxels in each dim
        fov : tuple
            FOV in meters in each dim
        dt : float
            dwell time in seconds
        B0 : float
            field strength in Tesla
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
        d = trj.shape[-1]
        trj_size = trj.shape[:-1]
        gamma_bar = 42.5774e6 # Hz / T
        assert d == 2 or d == 3
        assert d == len(fov)
        assert d == len(im_size)

        # Get gradient from trj
        trj = trj.type(torch.float32)
        if d == 2:
            trj = torch.cat((trj, trj[..., 0:1] * 0), dim=-1)
            fov += (1,)
        tup = (None,) * len(trj_size) + (slice(None),)
        fov_tensor = torch.tensor(fov).to(trj.device).type(trj.dtype)
        g = torch.diff(trj, dim=0) / (dt * gamma_bar * fov_tensor[tup])
        g = torch.cat((g, g[-1:]), dim=0)

        # Gen X Y Z grids
        grd = gen_grd(im_size, fov).to(trj.device).type(trj.dtype)
        if d == 2:
            grd = torch.concatenate((grd, grd[..., :1] * 0), dim=-1)
            
            if rotations is not None:
                rotations = torch.tensor(rotations, dtype=torch.float32, device=trj.device)
                thetas = torch.deg2rad(rotations)
                Rx = torch.tensor([
                    [1, 0, 0],
                    [0, torch.cos(thetas[0]), -torch.sin(thetas[0])],
                    [0, torch.sin(thetas[0]), torch.cos(thetas[0])]], 
                    device=trj.device)
                Ry = torch.tensor([
                    [torch.cos(thetas[1]), 0, torch.sin(thetas[1])],
                    [0, 1, 0],
                    [-torch.sin(thetas[1]), 0, torch.cos(thetas[1])]], 
                    device=trj.device)
                grd = (Ry @ Rx @ grd[..., None])[..., 0]
                g = (Ry @ Rx @ g[..., None])[..., 0]

            if z_ofs is not None:
                grd[..., 2] += z_ofs
        
        # Build phis and alphas
        phis = torch.zeros((4, *im_size), dtype=trj.dtype, device=trj.device)
        alphas = torch.zeros((4, *trj_size), dtype=trj.dtype, device=trj.device)
        X = grd[..., 0]
        Y = grd[..., 1]
        Z = grd[..., 2]
        phis[0] = X ** 2 + Y ** 2
        alphas[0] = (g[..., 2] ** 2) / 4
        phis[1] = Z ** 2
        alphas[1] = g[..., 0] ** 2 + g[..., 1] ** 2
        phis[2] = X * Z
        alphas[2] = g[..., 0] * g[..., 2]
        phis[3] = Y * Z
        alphas[3] = g[..., 1] * g[..., 2]
        alphas /= 2 * B0

        # Integral on alphas, gamma_bar to map T to phase
        alphas = torch.cumsum(alphas, dim=1) * dt * gamma_bar
        super(coco_imperfection, self).__init__(phis, alphas, L, method, interp_type, verbose)

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