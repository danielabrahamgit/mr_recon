import torch
import torch.nn as nn

from typing import Optional
from einops import rearrange
from fast_pytorch_kmeans import KMeans

class field_handler: 

    def __init__(self,
                 alphas: torch.Tensor,
                 phis: torch.Tensor):
        """
        Represents field imperfections of the form 

        phi(r, t) = sum_i phis_i(r) * alphas_i(t)

        where the signal in the presence of field imperfections is 
        
        s(t) = int_r m(r) e^{-j 2pi k(t) * r} e^{-j 2pi phi(r, t)} dr

        Parameters:
        -----------
        alphas : torch.Tensor <float32>
            basis coefficients with shape (*trj_shape, nbasis)
        phis : torch.Tensor <float32>
            spatial basis phase functions with shape (nbasis, N_{ndim-1}, ..., N_0)
        """

        msg = 'alphas and phis must be on the same device'
        assert alphas.device == phis.device, msg

        self.alphas = alphas
        self.phis = phis
        self.torch_dev = alphas.device
    
    def time_segments(self,
                          nseg: int) -> torch.Tensor:
        """
        Time segmentation takes the form

        e^{-j 2pi phi(r, t)} = sum_l e^{-j 2pi phi(r)_l } h_l(t)

        where 

        phi(r)_l = sum_i phis_i(r) * betas_i[l],
        
        and this function returns the betas

        Parameters:
        -----------
        nseg : int
            number of segments
        
        Returns:
        --------
        betas : torch.Tensor
            basis coefficients with shape (nseg, nbasis)
        """

        # Cluster the alpha coefficients
        alphas_flt = rearrange(self.alphas, '... nbasis -> (...) nbasis')
        alphas_flt = alphas_flt.to(self.torch_dev)
        kmeans = KMeans(n_clusters=nseg,
                        mode='euclidean')
        idxs = kmeans.fit_predict(alphas_flt)
        betas = kmeans.centroids.cpu()
        
        return betas
        
    def calc_temporal_interpolators(self,
                                    betas: torch.Tensor,
                                    mode: Optional[str] = 'zero') -> torch.Tensor:
        """
        Calculates temporal interpolation coefficients h_l(t) 
        from the following model:
        
        e^{-j 2pi phi(r, t)} = sum_l e^{-j 2pi phi(r)_l } h_l(t)

        where 

        phi(r)_l = sum_i phis_i(r) * betas_i[l]

        Parameters:
        -----------
        betas : torch.tensor <float32>
            time segmentation coeffs with shape (nseg, nbasis)
        mode : str
            interpolator type from the list
                'zero' - zero order interpolator
                'linear' - linear interpolator 
                'lstsq' - least squares interpolator
        
        Returns:
        --------
        interp_coeffs : torch.tensor <float32>
            interpolation coefficients 'h_l(t)' with shape (*trj_shape, nseg)
        """



