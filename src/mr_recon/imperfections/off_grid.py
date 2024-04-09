import torch

from typing import Optional
from mr_recon.utils import gen_grd
from mr_recon.imperfections.exponential import exponential_imperfection

class off_grid_imperfection(exponential_imperfection):

    def __init__(self, 
                 im_size: tuple,
                 grid_deviations: torch.Tensor, 
                 L: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 verbose: Optional[bool] = True):
        """
        Parameters:
        -----------
        im_size : tuple
            The size of the image = (N_{ndim-1}, ... N_0)
        grid_deviations : torch.Tensor
            Deviations from the grid with shape (..., ndim) 
            model is trj = trj_grid + deviations
            trj is assumed to have been scaled from [-N_i/2, N_i/2] in ith axis
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
        # Make regular grid for phis, and alphas are just the deviations
        device = grid_deviations.device
        phis = gen_grd(im_size).to(device).type(torch.float32)
        alphas = grid_deviations.type(torch.float32)
        super(off_grid_imperfection, self).__init__(phis.moveaxis(-1, 0), 
                                                    alphas.moveaxis(-1, 0),
                                                    L, method, interp_type, verbose)

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

        return self._calc_svd_fourier_space()