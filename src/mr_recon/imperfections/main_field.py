import torch

from typing import Optional
from mr_recon.imperfections.lowrank_imperfection import exponential_imperfection

class main_field_imperfection(exponential_imperfection):

    def __init__(self, 
                 b0_map: torch.Tensor, 
                 trj_dims: tuple,
                 dt: float,
                 L: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 verbose: Optional[bool] = True):
        """
        Parameters:
        -----------
        b0_map : torch.Tensor
            B0 map in Hz with shape (*im_size)
        trj_dims : tuple
            The dimensions of the trajectory, not 1 in readout dir
        dt : float
            Time step in seconds
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
        phis = dt * b0_map[None, ...]
        k = -1
        for i in range(len(trj_dims)):
            if trj_dims[i] > 1:
                k = i
                break
        if k == -1:
            raise ValueError('No non-singleton dimensions found in trj_dims')
        nro = trj_dims[k]
        assert torch.prod(torch.tensor(trj_dims)) == nro, 'trj_dims must only have 1 non-singleton dimension'
        tup = (None,) * k + (slice(None),) + (None,) * (len(trj_dims) - k - 1)
        alphas = torch.arange(nro, device=b0_map.device, dtype=phis.dtype)[tup]
        alphas = alphas[None,]
        super(main_field_imperfection, self).__init__(phis, alphas, L, method, interp_type, verbose)

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

        return self._calc_svd_fourier_time()