import torch
import sigpy as sp

from typing import Optional, Union
from einops import rearrange, einsum
from fast_pytorch_kmeans import KMeans
from mr_recon.utils.func import (
    batch_iterator, 
    torch_to_np, 
    np_to_torch,
    sp_fft,
    sp_ifft,
    apply_window
)
        
def alpha_phi_from_b0(b0_map: torch.Tensor,
                      trj_shape: tuple,
                      dt: float) -> Union[torch.Tensor, torch.Tensor]:
    """
    Creates field_handler from b0 map

    Parameters:
    -----------
    b0_map : torch.Tensor <float32>
        field map in Hz with shape (N_{ndim-1}, ..., N_0)
    trj_shape : tuple <int>
        k-space trajectory shape, first dim is readout, ex: (nro, ..., d)
    dt : float
        dwell time in seconds
    
    Returns:
    --------
    alphas : torch.Tensor <float32>
        basis coefficients with shape (*trj_shape, 1)
    phis : torch.Tensor <float32>
        spatial basis phase functions with shape (1, N_{ndim-1}, ..., N_0)
    """

    # Consts
    nro = trj_shape[0]

    # Make alphas
    ro_lin = torch.arange(nro) / nro
    alphas = torch.zeros((*trj_shape[:-1], 1))
    tup = (slice(None),) + (None,) * (len(trj_shape) - 1)
    alphas[:, ...] = ro_lin[tup]

    # Make phi
    phis = b0_map[None, ...] * dt * nro 

    return alphas, phis

class field_handler: 

    def __init__(self,
                 alphas: torch.Tensor,
                 phis: torch.Tensor,
                 nseg: int,
                 mode: Optional[str] = 'zero'):
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
        nseg : int
            number of segments
        mode : str
            see _calc_temporal_interpolators
        """

        msg = 'alphas and phis must be on the same device'
        assert alphas.device == phis.device, msg

        # Store
        self.alphas = alphas
        self.phis = phis
        self.nseg = nseg
        self.torch_dev = alphas.device

        # Compute time segmentation
        self.betas = self._time_segments(nseg)
        self.interp_coeffs = self._calc_temporal_interpolators(self.betas, mode)

    def update_phis_size(self,
                         new_size: tuple):
        """
        Updates the spatial size of phis using fourier method

        Parameters:
        -----------
        new_size : tuple
            new size to resample to, has shape (M_{ndim-1}, ..., M_0)
        
        Saves/updates:
        --------------
        self.phis
        """

        # Rescale between -pi and pi
        ndim = self.phis.ndim - 1
        nbasis = self.phis.shape[0]
        mxs = torch.max(torch.abs(self.phis).reshape((nbasis, -1)), dim=-1)[0]
        tup = (slice(None),) + (None,) * ndim
        phis_rs = torch.pi * self.phis / mxs[tup]

        # FFT
        PHI = sp_fft(torch.exp(1j * phis_rs), dim=tuple(range(-ndim, 0)))

        # Zero pad/chop
        PHI_sp = torch_to_np(PHI)
        dev = sp.get_device(PHI_sp)
        with dev:
            oshape = (phis_rs.shape[0], *new_size)
            PHI_rs = sp.resize(PHI_sp, oshape)

        # Windowing
        PHI_rs = apply_window(PHI_rs, 'hamming')
        PHI_rs = np_to_torch(PHI_rs)

        # Recover phis
        phis_rs = sp_ifft(PHI_rs, dim=tuple(range(-ndim, 0)))
        phis_new = mxs[tup] * torch.angle(phis_rs) / torch.pi

        self.phis = phis_new

    def get_phase_maps(self,
                       segs: Optional[torch.Tensor] = slice(None)):
        """
        Gets phase maps for given segments

        Parameters:
        -----------
        segs : torch.Tensor <int>
            segments with shape (N,) where N <= self.nseg, 
            seg[i] in [0, ..., self.nseg]
        
        Returns:
        --------
        phase_maps : torch.tensor <complex64>
            phase maps with shape (len(segs), N_{ndim-1}, ..., N_0)
        """
        phase = einsum(self.phis, self.betas[segs], 'nbasis ..., nseg nbasis -> nseg ...')
        phase_maps = torch.exp(-2j * torch.pi * phase)

        return phase_maps.type(torch.complex64)

    def get_interp_ceoffs(self,
                          segs: Optional[torch.Tensor] = None):
        """
        Gets phase maps for given segments

        Parameters:
        -----------
        segs : torch.Tensor <int>
            segments with shape (N,) where N <= self.nseg, 
            seg[i] in [0, ..., self.nseg]
        
        Returns:
        --------
        interp_coeffs : torch.tensor <float32>
            interpolation coefficients 'h_l(t)' with shape (*trj_shape, nseg)
        """
        assert segs.device == self.torch_dev, 'segs must be on same device as phis'
        return self.interp_coeffs[..., segs]

    def _time_segments(self,
                       nseg: int,
                       method: Optional['str'] = 'cluster') -> torch.Tensor:
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
        method : str
            selects the time segmentation method from:
            'cluster' - uses k-means to optimally find time-segments
            'uniform' - uniformly spaced time segments
        
        Returns:
        --------
        betas : torch.Tensor <float32>
            basis coefficients with shape (nseg, nbasis)
        """

        # Flatten alpha coeffs
        alphas_flt = rearrange(self.alphas, '... nbasis -> (...) nbasis')

        # Cluster the alpha coefficients
        if method == 'cluster':
            if (self.torch_dev.index == -1) or (self.torch_dev.index is None):
                kmeans = KMeans(n_clusters=nseg,
                                    mode='euclidean')
                idxs = kmeans.fit_predict(alphas_flt)
            else:
                with torch.cuda.device(self.torch_dev):
                    kmeans = KMeans(n_clusters=nseg,
                                    mode='euclidean')
                    idxs = kmeans.fit_predict(alphas_flt)
            self.idxs = idxs.reshape(self.alphas.shape[:-1])
            betas = kmeans.centroids

        # Uniformly spaced time segments
        else:
            # TODO FIXME For higher dims, only works for nbasis = 1
            nbasis = alphas_flt.shape[-1]
            betas = torch.zeros((nseg, nbasis), dtype=torch.float32, device=alphas_flt.device)
            for i in range(nbasis):
                lin = torch.linspace(start=alphas_flt[:, i].min(), 
                                     end=alphas_flt[:, i].max(), 
                                     steps=nseg + 1, 
                                     device=alphas_flt.device)
                betas[:, i] = (lin[:-1] + lin[1:]) / 2
        return betas.type(torch.float32)
    
    def _calc_temporal_interpolators(self,
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

        # alphas -> (*trj_shape, nbasis)
        # betas  -> (nseg,       nbasis)
        # hl(t)  -> (*trj_shape, nseg)

        # Consts
        assert self.torch_dev == betas.device
        nseg = betas.shape[0]
        trj_shape = self.alphas.shape[:-1]
        
        # Empty return coefficients
        interp_coeffs = torch.zeros((*trj_shape, nseg), dtype=torch.float32, device=self.torch_dev)

        # Least squares interpolator
        if 'lstsq' in mode:

            # TODO OptimizeME with quantization/histograms

            # Prep A, B matrices
            alphas_flt = rearrange(self.alphas, '... nbasis -> (...) nbasis').type(torch.float32)
            phis_flt = rearrange(self.phis, 'nbasis ... -> (...) nbasis').type(torch.float32)
            betas = self.betas.type(torch.float32)
            AHA = torch.zeros((nseg, nseg), 
                              dtype=torch.complex64, device=self.torch_dev)
            AHb = torch.zeros((nseg, alphas_flt.shape[0]), 
                              dtype=torch.complex64, device=self.torch_dev)
            batch_size = 2 ** 8
            for n1, n2 in batch_iterator(phis_flt.shape[0], batch_size):

                # Accumulate AHA
                A_batch = einsum(phis_flt[n1:n2, :], betas, 
                                 'B nbasis, nseg nbasis -> B nseg')
                A_batch = torch.exp(-2j * torch.pi * A_batch)
                AHA += A_batch.H @ A_batch

                # Accumulate AHb
                B_batch = einsum(phis_flt[n1:n2, :], alphas_flt,
                                 'B nbasis, T nbasis -> B T')
                B_batch = torch.exp(-2j * torch.pi * B_batch)
                AHb += A_batch.H @ B_batch

            # Solve for x = (AHA)^{-1} AHb, sigpy least squares is OP
            AHA_cp = torch_to_np(AHA)
            AHb_cp = torch_to_np(AHb)
            dev = sp.get_device(AHA_cp)
            with dev:
                x = dev.xp.linalg.lstsq(AHA_cp, AHb_cp, rcond=None)[0]
            x = np_to_torch(x)
            
            # Reshape (nseg, T)
            interp_coeffs = x.T.reshape((*trj_shape, nseg))
            interp_coeffs = rearrange(x, 'nseg (nro npe ntr) -> nro npe ntr nseg',
                                      nro=trj_shape[0], npe=trj_shape[1], ntr=trj_shape[2])
        
        # Linear interpolator
        elif 'lin' in mode:
            raise NotImplementedError
        
        # Zero order hold/nearest interpolator
        else:

            # Find closest points
            tup = (slice(None),) + (None,) * (self.alphas.ndim - 1) + (slice(None),)
            inds = torch.argmin(
                torch.linalg.norm(self.alphas[None, ...] - betas[tup], dim=-1),
                dim=0) # *trj_shape -> values in [0, ..., nseg-1]
            
            # Indicator function
            for i in range(nseg):
                interp_coeffs[..., i] = 1.0 * (inds == i)

        return interp_coeffs.type(torch.complex64)
  