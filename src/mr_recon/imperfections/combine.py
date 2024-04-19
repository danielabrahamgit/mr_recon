"""
Combines multiple general imperfections into one
"""
import torch

from typing import Optional
from mr_recon.imperfections.imperfection import imperfection

class combined_imperfections(imperfection):
    """
    Combines multiple imperfections into one
    """

    def __init__(self, 
                 imperfections: list[imperfection],
                 torch_dev: Optional[torch.device] = torch.device('cpu')):
        """
        Parameters:
        -----------
        imperfections : list
            A list of imperfections to combine
        torch_dev : Optional[torch.device]
            The device to use for torch operations
        """
        self.imperfections = imperfections
        self.L = torch.prod(torch.tensor([imp.L for imp in imperfections])).item()

        self.im_size = imperfections[0].im_size
        self.trj_size = imperfections[0].trj_size

        # Useful for batch processing later
        self.Ls = torch.tensor([imp.L for imp in imperfections])
        self.all_inds = torch.cartesian_prod(*[torch.arange(l, device=torch_dev) for l in self.Ls]) # (self.L, num_imperf)

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

    def _l_slice_helper(self,
                        ls: Optional[torch.Tensor] = slice(None)):
        """
        Helper for apply_spatial/temporal/adjoint functions.

        Parameters:
        -----------
        ls : Optional[torch.Tensor]
            Slices the lowrank dimension, has size at most L
        
        Returns:
        --------
        start : torch.Tensor
            The start of the valid slice
        stop : torch.Tensor
            The end of the valid slice
        valid : torch.Tensor
            A boolean tensor indicating if the slice is valid
        """

        if type(ls) is slice:
            l1 = ls.start
            l2 = ls.stop - 1
        elif type(ls) is torch.Tensor:
            l1 = ls[0]
            l2 = ls[-1]
        start = torch.maximum(self.Cs * 0, l1 - self.Cs)
        stop = torch.minimum(self.Ls - 1, l2 - self.Cs)
        valid = (start < self.Ls) * (stop >= 0)
        start = start * valid
        stop = stop * valid

        return start, stop, valid
    
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
        y_all : torch.Tensor
            The images with all spatial functions applied with shape (..., len(ls), *im_size)
        """
        inds_batch = self.all_inds[ls]
        x_ones = torch.ones_like(x)
        y_all = None
        for k, imp in enumerate(self.imperfections):
            if k == 0:
                y_all = imp.apply_spatial(x, inds_batch[:, k])
            else:
                y_all *= imp.apply_spatial(x_ones, inds_batch[:, k])
        return y_all
        
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
        x_all : torch.Tensor
            The image with all spatial adjoint transforms applied with shape (..., *im_size)
        """
        inds_batch = self.all_inds[ls]
        im_and_batch_size = y.shape[:-len(self.im_size)-1] + y.shape[-len(self.im_size):]
        y_ones = torch.ones(im_and_batch_size, device=y.device, dtype=y.dtype)
        expanded = None
        for k, imp in enumerate(self.imperfections):
            if k == 0:
                expanded = imp.apply_spatial(y_ones, inds_batch[:, k])
            else:
                expanded *= imp.apply_spatial(y_ones, inds_batch[:, k])
        x_all = (expanded.conj() * y).sum(dim=-len(self.im_size)-1)
        return x_all

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
        y_all : torch.Tensor
            The output of all combined temporal data with shape (..., *trj_size)
        """
        inds_batch = self.all_inds[ls]
        trj_and_batch_size = x.shape[:-len(self.trj_size)-1] + x.shape[-len(self.trj_size):]
        x_ones = torch.ones(trj_and_batch_size, device=x.device, dtype=x.dtype)
        expanded = None
        for k, imp in enumerate(self.imperfections):
            if k == 0:
                expanded = imp.apply_temporal_adjoint(x_ones, inds_batch[:, k])
            else:
                expanded *= imp.apply_temporal_adjoint(x_ones, inds_batch[:, k])
        return (expanded.conj() * x).sum(dim=-len(self.trj_size)-1)
    
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
        x_all : torch.Tensor
            The output of all temporal data with shape (..., len(ls), *trj_size)
        """
        inds_batch = self.all_inds[ls]
        x_all = None
        y_ones = torch.ones_like(y)
        for k, imp in enumerate(self.imperfections):
            if k == 0:
                x_all = imp.apply_temporal_adjoint(y, inds_batch[:, k])
            else:
                x_all *= imp.apply_temporal_adjoint(y_ones, inds_batch[:, k])
        return x_all
    
