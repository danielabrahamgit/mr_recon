import torch

from typing import Optional
from einops import rearrange
from mr_recon.imperfections.exponential import exponential_imperfection

class main_field_imperfection(exponential_imperfection):

    def __init__(self, 
                 b0_map: torch.Tensor, 
                 trj_size: tuple,
                 ro_dim: int,
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
        trj_size : tuple
            The size of the trajectory
        ro_dim : int
            specifies the readout dim from from trj_size
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
        phis = b0_map[None, ...] * dt
        nro = trj_size[ro_dim]
        tup = (None,) * ro_dim + (slice(None),) + (None,) * (len(trj_size) - ro_dim - 1)
        alphas = torch.arange(nro, device=b0_map.device, dtype=phis.dtype)[tup]
        alphas = alphas[None,]
        self.prod1 = torch.prod(torch.tensor(trj_size)[ro_dim:]).item()
        self.prod2 = torch.prod(torch.tensor(trj_size)[ro_dim+1:]).item()
        self.orig_trj_size = trj_size
        super(main_field_imperfection, self).__init__(phis, alphas, L, method, interp_type, verbose)

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
        
        # Flatten spatial dims
        x = x.flatten(start_dim=-len(self.im_size))[..., r_inds]

        # Re broadcast for readout dimension
        t_inds_new = (t_inds % self.prod1 - t_inds % self.prod2) // self.prod2

        if lowrank:
            h = self.apply_temporal_adjoint().reshape((self.L, -1))[:, t_inds_new]
            b = self.apply_spatial().reshape((self.L, -1))[:, r_inds]
            xt = torch.sum(b * h, dim=0) * x
        else:
            alphas = self.alphas.reshape((self.B, -1))[:, t_inds_new]
            phis = self.phis.reshape((self.B, -1))[:, r_inds]
            xt = torch.exp(-2j * torch.pi * torch.sum(alphas * phis, dim=0)) * x
        return xt

    def get_network_features(self) -> torch.Tensor:
        """
        Relevant network features are the alpha coefficients and the hl functions

        Returns:
        --------
        features : torch.Tensor
            The features of the imperfection with shape (*trj_size, nfeat)
        """
        # features = super().get_network_features()
        if self.method == 'ts':
            features = self.alphas.moveaxis(0, -1) - self.alpha_clusters[:, 0]
        elif self.method == 'svd':
            features = self.alphas.moveaxis(0, -1) - self.alphas.max() / 2 
            features /= self.L
        features *= self.L / self.alphas.max()
        features = features.expand(tuple(self.orig_trj_size) + (features.shape[-1],))
        return features

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