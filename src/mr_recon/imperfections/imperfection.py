import torch

from typing import Optional

class imperfection:
    """
    Base class for imperfections in MRI. All Imperfections
    take the form:

    b(t) = int_r m(r) T(t){s(r)} e^{-j 2pi k(t) r} dr

    Where T(t) is a spatio-temporal transform describing the imperfection,
    which is applied to the sens maps independently per channel.

    The lowrank model is given by:

    T(t){x(r)} = sum_{l=1}^L h_l(t) T_l{x(r)}
    """

    def __init__(self, 
                 L: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 verbose: Optional[bool] = True):
        """
        Parameters:
        -----------
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
        self.L = L
        self.method = method.lower()
        self.interp_type = interp_type.lower()
        self.verbose = verbose
        
        # Should bepopulated by children of this class
        if not hasattr(self, 'im_size'):
            self.im_size = None
        if not hasattr(self, 'trj_size'):
            self.trj_size = None

        if self.method not in ['ts', 'svd']:
            raise ValueError('method must be one of ["ts", "svd"]')
        
        if self.interp_type not in ['zero', 'linear', 'lstsq']:
            raise ValueError('interp_type must be one of ["zero", "linear", "lstsq"]')
        
        self._low_dim_decomp()
    
    def _low_dim_decomp(self):
        """
        Decomposes the lowrank model into spatial and temporal components
        """
        if self.method == 'ts':
            self._calc_time_segmentation()
        else:
            self.spatial_funcs, self.temporal_funcs = self._calc_svd()

    def _calc_time_segmentation(self):
        """
        Computes necessary information for time segmented based splitting
        """
        raise NotImplementedError

    def _calc_svd(self):
        """
        Computes necessary information for SVD based splitting
        """
        raise NotImplementedError
    
    def get_network_features(self) -> torch.Tensor:
        """
        Every imperfection will be characterized by some important 
        features. For example, Motion imperfections naturally should 
        return rigid body motion parameters, etc.

        Returns:
        --------
        features : torch.Tensor
            The features of the imperfection
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def apply_spatial(self,
                      x: torch.Tensor,
                      ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        """
        Applies the spatial part of the lowrank model to image
        y(r, l) = T_l{x(r)}

        Parameters:
        -----------
        x : torch.Tensor
            The input image with shape (..., *im_size)
        ls : Optional[torch.Tensor]
            Slices the lowrank dimension, has size at most L
        
        Returns:
        --------
        y : torch.Tensor
            The images with spatial transforms applied with shape (..., len(ls), *im_size)
        """
        raise NotImplementedError
    
    def apply_spatial_adjoint(self,
                              y: torch.Tensor,
                              ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        """
        Applies the adjoint of the spatial part of the lowrank model to images
        x(r) = T_l^H{y(r, l)}

        Parameters:
        -----------
        y : torch.Tensor
            The input images with shape (..., len(ls), *im_size)
        ls : Optional[torch.Tensor]
            Slices the lowrank dimension
        
        Returns:
        --------
        x : torch.Tensor
            The image with spatial adjoint transforms applied with shape (..., *im_size)
        """
        raise NotImplementedError

    def apply_temporal(self,
                       x: torch.Tensor,
                       ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        """
        Applies the temporal part of the lowrank model to temporal data
        y(t) = sum_l h_l(t) * x_l(t)

        Parameters:
        -----------
        x : torch.Tensor
            The input temporal data with shape (..., len(ls), *trj_size)
        ls : Optional[torch.Tensor]
            Slices the lowrank dimension
        
        Returns:
        --------
        y : torch.Tensor
            The output temporal data with shape (..., *trj_size)
        """
        raise NotImplementedError
    
    def apply_temporal_adjoint(self,
                               y: torch.Tensor,
                               ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        """
        Applies the adjoint of the temporal part of the lowrank model to input
        x_l(t) = conj(h_l(t)) * y(t)

        Parameters:
        -----------
        y : torch.Tensor
            The input temporal data with shape (..., *trj_size)
        ls : Optional[torch.Tensor]
            Slices the lowrank dimension
        
        Returns:
        --------
        x : torch.Tensor
            The output temporal data with shape (..., len(ls), *trj_size)
        """
        raise NotImplementedError
    
