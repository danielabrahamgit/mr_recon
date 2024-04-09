import torch

from typing import Optional
from mr_recon.imperfections.imperfection import imperfection

class combined_imperfections(imperfection):
    """
    Combines multiple imperfections into one
    """

    def __init__(self, 
                 imperfections: list[imperfection]):
        """
        Parameters:
        -----------
        imperfections : list
            A list of imperfections to combine
        """
        self.imperfections = imperfections
        self.L = sum([imp.L for imp in imperfections])

        # Useful for batch processing later
        self.ls = torch.tensor([imp.L for imp in imperfections])
        self.ls_cmsm = torch.cumsum(self.ls, dim=0)

    def _calc_time_segmentation(self):
        """
        Computes necessary information for time segmented based splitting
        """
        for imp in self.imperfections:
            imp._calc_time_segmentation()

    def _calc_svd(self):
        """
        Computes necessary information for SVD based splitting
        """
        for imp in self.imperfections:
            imp._calc_svd()

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
        return torch.cat([imp.get_network_features() for imp in self.imperfections], dim=0)
    
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
        # TODO THINK!!!
        for imp in self.imperfections:
            x = imp.apply_spatial(x, ls)
        
    
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
    
