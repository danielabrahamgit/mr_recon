import torch
import sigpy as sp

from tqdm import tqdm
from typing import Optional, Union
from einops import rearrange, einsum
from mr_recon.utils import (
    torch_to_np, 
    np_to_torch,
    apply_window,
    quantize_data,
    batch_iterator
)
from mr_recon.fourier import fft, ifft
from mr_recon.algs import lin_solve, svd_power_method_tall

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
            torch.ones_like(x), # 0
            x, # 1
            y, # 2 
            z, # 3
            x * y, # 4
            z * y, # 5
            3 * z2 - (x2 + y2 + z2), # 6
            x * z, # 7
            x2 - y2, # 8
            3 * y * x2 - y3, # 9
            x * y * z, # 10
            (5 * z2 - (x2 + y2 + z2)) * y, # 11
            5 * z3 - 3 * z * (x2 + y2 + z2), # 12
            (5 * z2 - (x2 + y2 + z2)) * x, # 13
            z * x2 - z * y2, # 14
            x3 - 3 * x * y2 # 15
        ], dim=0)
        
def alpha_phi_from_b0(b0_map: torch.Tensor,
                      trj_size: tuple,
                      dt: float) -> Union[torch.Tensor, torch.Tensor]:
    """
    Creates field_handler from b0 map

    Parameters:
    -----------
    b0_map : torch.Tensor <float32>
        field map in Hz with shape (N_{ndim-1}, ..., N_0)
    trj_size : tuple <int>
        k-space trajectory shape, first dim is readout, ex: (nro, ..., d)
    dt : float
        dwell time in seconds
    
    Returns:
    --------
    alphas : torch.Tensor <float32>
        basis coefficients with shape (*trj_size, 1)
    phis : torch.Tensor <float32>
        spatial basis phase functions with shape (1, N_{ndim-1}, ..., N_0)
    """

    # Consts
    nro = trj_size[0]

    # Make alphas
    dev = b0_map.device
    ro_lin = torch.arange(nro, device=dev) * dt
    alphas = torch.zeros((*trj_size[:-1], 1), device=dev)
    tup = (slice(None),) + (None,) * (len(trj_size) - 1)
    alphas[:, ...] = ro_lin[tup]

    # Make phi
    phis = b0_map[None, ...]

    return alphas, phis

def alpha_phi_from_skope(skope_data: torch.Tensor,
                         im_size: tuple,
                         fov: tuple,
                         skope_inds: Optional[torch.Tensor] = None) -> Union[torch.Tensor, torch.Tensor]:
    """
    Constructs alphas and phis from raw skope data

    Parameters:
    -----------
    skope_data : torch.Tensor <float32>
        this is technically alphas, has shape (*trj_size, nbasis)
    im_size : tuple
        num voxels in each dim
    fov : tuple
        FOV in meters in each dim
    skope_inds : torch.Tensor <int>
        specifies which skope terms are in the nbasis dimension.
        For example, skope_terms = [0, 2, 3] with nbasis = 3 will 
        incorporate 0 order, Y, and Z skope terms.
        defaults to range(nbasis)
    
    Returns:
    --------
    alphas : torch.Tensor <float32>
        temoral coefficients with shape (*trj_size, nbasis)
    phis : torch.Tensor <float32>
        spatial basis phase functions with shape (nbasis, *im_size)
    """

    # Consts
    d = len(im_size)
    nbasis = skope_data.shape[-1]
    alphas = skope_data / (2 * torch.pi)

    if skope_inds is None:
        skope_inds = torch.arange(nbasis)

    # Gen X Y Z grids
    assert nbasis == len(skope_inds)
    assert d == 2 or d == 3
    assert d == len(fov)
    for i in range(d):
        assert im_size[i] % 2 == 0
    x = fov[0] * torch.arange(-(im_size[0]//2), im_size[0]//2) / im_size[0]
    y = fov[1] * torch.arange(-(im_size[1]//2), im_size[1]//2) / im_size[1]
    if d == 2:
        X, Y = torch.meshgrid(x, y, indexing='ij')
        Z = X * 0
    else:
        z = fov[2] * torch.arange(-(im_size[2]//2), im_size[2]//2) / im_size[2]
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Phi basis functions
    skope_inds = skope_inds.to(alphas.device)
    all_bases = bases(X, Y, Z).to(alphas.device)
    phis = all_bases[skope_inds]

    # # Scale
    # for i in range(nbasis):
    #     mx = phis[i].abs().max()
    #     phis[i, ...] /= (mx + 1e-8)
    #     alphas[..., i] *= mx

    return alphas, phis

def alpha_phi_from_maxwell(trj: torch.Tensor,
                           im_size: tuple,
                           fov: tuple,
                           dt: float,
                           B0: float,
                           z_ofs: Optional[float],
                           rotations: Optional[tuple]) -> Union[torch.Tensor, torch.Tensor]:
    """
    Constructs alphas and phis from analytic maxwell field model.
    Function of gradient waveform only
    
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
            
    Returns:
    --------
    alphas : torch.Tensor <float32>
        temoral coefficients with shape (*trj_size, nbasis)
    phis : torch.Tensor <float32>
        spatial basis phase functions with shape (nbasis, *im_size)
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
    fov = torch.tensor(fov).to(trj.device).type(trj.dtype)
    g = torch.diff(trj, dim=0) / (dt * gamma_bar * fov[tup])
    g = torch.cat((g, g[-1:]), dim=0)

    # Gen X Y Z grids
    for i in range(d):
        assert im_size[i] % 2 == 0
    lins = [
        torch.arange(-(im_size[i]//2), im_size[i]//2, 
                     device=trj.device, dtype=torch.float32) / (im_size[i]) * fov[i]
        for i in range(d)
        ]
    grds = torch.meshgrid(*lins, indexing='ij')
    grd = torch.concatenate(
        [g[..., None] for g in grds], axis=-1)
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
    alphas = torch.zeros((*trj_size, 4), dtype=trj.dtype, device=trj.device)
    X = grd[..., 0]
    Y = grd[..., 1]
    Z = grd[..., 2]
    phis[0] = X ** 2 + Y ** 2
    alphas[..., 0] = (g[..., 2] ** 2) / 4
    phis[1] = Z ** 2
    alphas[..., 1] = g[..., 0] ** 2 + g[..., 1] ** 2
    phis[2] = X * Z
    alphas[..., 2] = g[..., 0] * g[..., 2]
    phis[3] = Y * Z
    alphas[..., 3] = g[..., 1] * g[..., 2]
    alphas /= 2 * B0

    # Integral on alphas, gamma_bar to map T to phase
    alphas = torch.cumsum(alphas, dim=0) * dt * gamma_bar

    return alphas, phis

class field_handler: 

    def __init__(self,
                 alphas: torch.Tensor,
                 phis: torch.Tensor,
                 nseg: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 quant_type: Optional[str] = 'uniform',
                 verbose: Optional[bool] = True):
        """
        Represents field imperfections of the form 

        phi(r, t) = sum_i phis_i(r) * alphas_i(t)

        where the signal in the presence of field imperfections is 
        
        s(t) = int_r m(r) e^{-j 2pi k(t) * r} e^{-j 2pi phi(r, t)} dr

        Parameters:
        -----------
        alphas : torch.Tensor <float32>
            basis coefficients with shape (*trj_size, nbasis)
        phis : torch.Tensor <float32>
            spatial basis phase functions with shape (nbasis, *im_size)
        nseg : int
            number of segments
        method : str
            'ts' - time segmentation
            'fs' - frequency segmentatoin/mfi
            'svd' - SVD based splitting
        interp_type : str
            'zero' - zero order interpolator
            'linear' - linear interpolator 
            'lstsq' - least squares interpolator
        quant_type : str
            'cluster' - uses k-means to optimally find centers
            'uniform' - uniformly spaced bins
        verbose : bool
            toggles print statements
        """

        msg = 'alphas and phis must be on the same device'
        assert alphas.device == phis.device, msg
        assert alphas.shape[-1] == phis.shape[0]

        # Store
        self.alphas = alphas
        self.phis = phis
        self.nseg = nseg
        self.torch_dev = alphas.device
        self.method = method.lower()
        self.interp_type = interp_type.lower()
        self.trj_size = alphas.shape[:-1]
        self.im_size = phis.shape[1:]
        self.nbasis = phis.shape[0]
        self.verbose = verbose
        self.spatial_batch_size = 2 ** 8

        # Store flattened version without more memory
        self.phis_flt = rearrange(phis, 'nbasis ... -> nbasis (...)')
        self.alphas_flt = rearrange(alphas, '... nbasis -> (...) nbasis')

        if 'ts' in method.lower():
            # Compute time segmentation
            self.betas = quantize_data(alphas.reshape((-1, self.nbasis)), 
                                             nseg, method=quant_type)
            self.temporal_funcs = self._calc_temporal_interpolators(self.betas, interp_type)
            self.temporal_funcs_flt = rearrange(self.temporal_funcs, '... nseg -> nseg (...)')
        elif 'fs' in method.lower():
            self.thetas = quantize_data(phis.reshape((self.nbasis, -1)).T, 
                                              nseg, method=quant_type)
            self.spatial_funcs = self._calc_spatial_interpolators(self.thetas, interp_type)
            self.spatial_funcs_flt = rearrange(self.spatial_funcs, 'nseg ... -> nseg (...)')
        elif 'svd' in method.lower():
            self.spatial_funcs, self.temporal_funcs = self._calc_svd(operator_method=True, spatial_batch_size=2 ** 15)
            self.spatial_funcs_flt = rearrange(self.spatial_funcs, 'nseg ... -> nseg (...)')
            self.temporal_funcs_flt = rearrange(self.temporal_funcs, '... nseg -> nseg (...)')
        else:
            raise NotImplementedError

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
        ndim = len(self.im_size)
        nbasis = self.nbasis
        mxs = torch.max(torch.abs(self.phis).reshape((nbasis, -1)), dim=-1)[0]
        tup = (slice(None),) + (None,) * ndim
        phis_rs = torch.pi * self.phis / mxs[tup]

        # FFT
        PHI = fft(torch.exp(1j * phis_rs), dim=tuple(range(-ndim, 0)))

        # Zero pad/chop
        PHI_sp = torch_to_np(PHI)
        dev = sp.get_device(PHI_sp)
        with dev:
            oshape = (phis_rs.shape[0], *new_size)
            PHI_rs = sp.resize(PHI_sp, oshape)

        # Windowing
        PHI_rs = apply_window(PHI_rs, PHI_rs.ndim - 1, 'hamming')
        PHI_rs = np_to_torch(PHI_rs)

        # Recover phis
        phis_rs = ifft(PHI_rs, dim=tuple(range(-ndim, 0)))
        phis_new = mxs[tup] * torch.angle(phis_rs) / torch.pi

        self.phis = phis_new

    def topelitz_weights(self):

        # Zero order interp has O(L) toeplitz weights
        if 'svd' not in self.method and self.interp_type == 'zero':
            weights = self.get_temporal_funcs()
            weights = (weights * weights.conj())[None,]
        # Other interp has O(L^2) toeplitz weights
        else:
            weights = self.get_temporal_funcs()
            weights = weights[None,].conj() * weights[:, None]
        return weights

    def get_spatial_funcs(self,
                          segs: Optional[torch.Tensor] = slice(None),
                          r_inds: Optional[torch.Tensor] = None):
        """
        Gets spatial basis functons at given segments

        Parameters:
        -----------
        segs : torch.Tensor <int>
            segments with shape (N,) where N <= self.nseg, 
            seg[i] in [0, ..., self.nseg]
        r_inds : torch.Tensor <int>
            Optionally selects subset of spatial inds.
            Spatial indices are for flattened spatial terms, shape is (Nr,)
        
        Returns:
        --------
        spatial_funcs : torch.tensor <complex64>
            spatial basis functions with shape (len(segs), N_{ndim-1}, ..., N_0)
            OR, if r_inds is given
            spatial basis functions with shape (len(segs), len(r_inds))
        """

        if 'ts' in self.method:
            if r_inds is None:
                phase = einsum(self.phis, self.betas[segs], 'nbasis ..., nseg nbasis -> nseg ...')
            else:
                phase = einsum(self.phis_flt[:, r_inds], self.betas[segs], 'nbasis R, nseg nbasis -> nseg R')
            phase_maps = torch.exp(-2j * torch.pi * phase)
            spatial_funcs = phase_maps.type(torch.complex64)
        elif 'fs' in self.method or 'svd' in self.method:
            if r_inds is None:
                spatial_funcs = self.spatial_funcs[segs]
            else:
                spatial_funcs = self.spatial_funcs_flt[segs, r_inds]
        else:
            raise NotImplementedError

        return spatial_funcs

    def get_temporal_funcs(self,
                           segs: Optional[torch.Tensor] = slice(None),
                           t_inds: Optional[torch.Tensor] = None):
        """
        Gets temporal functions at given segments

        Parameters:
        -----------
        segs : torch.Tensor <int>
            segments with shape (N,) where N <= self.nseg, 
            seg[i] in [0, ..., self.nseg]
        t_inds : torch.Tensor <int>
            Optionally selects subset of temporal inds.
            Temporal indices are for flattened temporal terms, shape is (Nt,)
        
        Returns:
        --------
        temporal_funcs : torch.tensor <float32>
            temporal functions 'h_l(t)' with shape (*trj_size, len(segs))
            OR, if t_inds is given
            temporal functions 'h_l(t)' with shape (len(t_inds), len(segs))
        """

        if 'ts' in self.method or 'svd' in self.method:
            if t_inds is None:
                temporal_funcs = self.temporal_funcs[..., segs]
            else:
                temporal_funcs = self.temporal_funcs_flt[t_inds, segs]
        elif 'fs' in self.method:
            if t_inds is None:
                phase = einsum(self.alphas, self.thetas[segs], 
                            '... nbasis, nseg nbasis -> ... nseg')
            else:
                phase = einsum(self.alphas_flt[t_inds], self.thetas[segs], 
                            'T nbasis, nseg nbasis -> T nseg')
            phase = torch.exp(-2j * torch.pi * phase)
            temporal_funcs = phase.type(torch.complex64)
        else:
            raise NotImplementedError

        return temporal_funcs

    def get_grog_features(self):

        alpha_vecs = self.alphas # *trj_size nbasis
        temporal_functions = self.get_temporal_funcs() # *trj_size nseg
        grog_features = torch.hstack([alpha_vecs, temporal_functions.real])
        imag_nrm = torch.sum(torch.abs(temporal_functions.imag))
        if imag_nrm > 1e-9:
            grog_features = torch.hstack([grog_features, temporal_functions.imag])
        return grog_features

    def phase_est(self,
                  r_inds: torch.Tensor,
                  t_inds: torch.Tensor,
                  lowrank: Optional[bool] = False) -> torch.Tensor:
        """
        Estimates phase at given spatial/temporal points.
         
        If lowrank:

            e^{-j 2pi phi(r, t)} = sum_l b_l(r) * h_l(t)
        
        Otherwise

            e^{-j 2pi phi(r, t)} = e^{-j 2pi sum_k phis_k(r) * alphas_k(t)}

        Parameters:
        -----------
        r_inds : torch.Tensor
            spatial indices for flattened spatial terms, shape is (N,)
        t_inds : tuple
            temporal indices for flattened temporal terms, shape is (N,)
        lowrank : bool
            if True, uses low rank phase model
        
        Returns:
        --------
        phase_est : torch.Tensor
            estimated phase with shape (N, )
        """

        if lowrank:
            b = self.get_spatial_funcs(r_inds)
            h = self.get_temporal_funcs(t_inds)
            return einsum (b, h, 'nseg N, N nseg -> N')
        else:
            phi_chunk = self.phis_flt[:, r_inds]
            alpha_chunk = self.alphas_flt[t_inds, :]
            phase_chunk = einsum(phi_chunk, alpha_chunk, 'nbasis nr, nt nbasis -> nr nt')
            phase_est = torch.exp(-2j * torch.pi * phase_chunk)

        return phase_est

    def phase_est_slice(self,
                  r_slice: tuple, 
                  t_slice: tuple,
                  lowrank: Optional[bool] = False) -> torch.Tensor:
        """
        Estimates phase at given spatial/temporal points.
         
        If lowrank:

            e^{-j 2pi phi(r, t)} = sum_l b_l(r) * h_l(t)
        
        Otherwise

            e^{-j 2pi phi(r, t)} = e^{-j 2pi sum_k phis_k(r) * alphas_k(t)}

        Parameters:
        -----------
        r_slice : tuple
            spatial indices with length len(im_size)
        t_slice : tuple
            temporal indices with length len(trj_size)
        lowrank : bool
            if True, uses low rank phase model
        
        Returns:
        --------
        phase_est : torch.Tensor
            estimated phase with shape (*im_size[r_slice], *trj_shape[t_slice])
        """

        if lowrank:
            segs = torch.ones(1, dtype=torch.long, device=self.torch_dev)
            phase_est = None
            for i in range(self.nseg):
                
                # Get spatial temporal basis funcs 
                b = self.get_spatial_funcs(segs * i)[0][r_slice] # (*im_size)
                h = self.get_temporal_funcs(segs * i)[..., 0][t_slice] # (*trj_size)
                tup_b = (slice(None),) * b.ndim + (None,) * h.ndim
                tup_h = (None,) * b.ndim + (slice(None),) * h.ndim

                # accumulate
                if phase_est is None:
                    phase_est = b[tup_b] * h[tup_h]
                else:
                    phase_est += b[tup_b] * h[tup_h]
        else:
            phase_est = None
            for i in range(self.nbasis):

                # Get spatial temporal basis funcs 
                phi = self.phis[i][r_slice] # (*im_size)
                alpha = self.alphas[..., i][t_slice] # (*trj_size)
                tup_p = (slice(None),) * phi.ndim + (None,) * alpha.ndim
                tup_a = (None,) * phi.ndim + (slice(None),) * alpha.ndim

                # accumulate
                if phase_est is None:
                    phase_est = phi[tup_p] * alpha[tup_a]
                else:
                    phase_est += phi[tup_p] * alpha[tup_a]
            
            phase_est = torch.exp(-2j * torch.pi * phase_est)

        return phase_est

    def _calc_svd(self,
                  operator_method: Optional[bool] = False,
                  assume_b0: Optional[bool] = True,
                  spatial_batch_size: Optional[int] = 2 ** 14) -> torch.Tensor:
        """
        Calculates spatial and temporal functions from SVD 
        of spatio-temporal matrix

        Parameters:
        -----------
        operator_method : bool
            if True, uses operator method to calculate SVD. 
            This is more memory efficient, but slower
            Otherwise constructs big spatiotemporal matrix
        assume_b0 : bool
            if True, assumes B0 case where alphas are 1d 
            and proportional to  time
        spatial_batch_size : int
            batch size over spatial terms
        
        Returns:
        --------
        spatial_funcs : torch.tensor <complex64>
            spatial basis functions with shape (nseg, *im_size)
        temporal_funcs : torch.tensor <complex64>
            temporal basis functions with shape (*trj_size, nseg)
        """

        if operator_method:
            def A(x):
                x = x.reshape((-1))
                nvox = torch.prod(torch.tensor(self.im_size)).item()
                y = torch.zeros((nvox,), dtype=torch.complex64, device=x.device)
                for n1, n2 in batch_iterator(nvox, spatial_batch_size):
                    y[n1:n2] = self.phase_est(slice(n1, n2), slice(None), lowrank=False) @ x
                return y.reshape(self.im_size)
            
            def AH(y):
                y = y.reshape((-1))
                nvox = torch.prod(torch.tensor(self.im_size)).item()
                ntrj = torch.prod(torch.tensor(self.trj_size)).item()
                x = torch.zeros((ntrj,), dtype=torch.complex64, device=y.device)
                for n1, n2 in batch_iterator(nvox, spatial_batch_size):
                    x += self.phase_est(slice(n1, n2), slice(None), lowrank=False).H @ y[n1:n2]
                return x.reshape(self.trj_size)
            if assume_b0:
                assert self.nbasis == 1

                import torchaudio

                def AHA(x):
                    # apply convolution is freq domain
                    orig_shape = x.shape
                    x = x.reshape((-1))
                    x_hat = torchaudio.functional.convolve(x, impulse_response, mode='same')
                    # x_hat = torch.fft.fft(x, n=2*len(x)) * impulse_FT
                    # x_hat = torch.fft.ifft(x_hat)#[:len(x)]
                    return x_hat.reshape(orig_shape)

                def AHA_plz(x):
                    return AH(A(x))
                
                # Find impulse response
                impulse_response = AH(torch.ones(self.im_size, dtype=torch.complex64, device=self.torch_dev))
                impulse_response = impulse_response.flatten()
                impulse_response = torch.hstack((torch.flip(impulse_response, dims=(0,)).conj(), impulse_response[1:]))

                delta = torch.zeros(self.trj_size, dtype=torch.complex64, device=self.torch_dev)
                delta[0, 0, 23] = 1
                impulse_response_AHA = AHA_plz(delta).flatten()

                impulse_response_AHA = impulse_response_AHA.cpu()
                impulse_response = impulse_response.cpu()
                
                import matplotlib.pyplot as plt
                from mr_recon.utils import normalize
                impulse_response_AHA = normalize(impulse_response_AHA, impulse_response, ofs=False, mag=True)
                plt.plot(impulse_response.abs().cpu(), label='AH(1)', alpha=0.4)
                plt.plot(impulse_response_AHA.abs().cpu(), label='AHA(d)', alpha=0.4)
                plt.legend()
                
                plt.show()
                quit()
            else:
                
                def AHA(s):
                    return AH(A(s))
            
            U, S, Vh = svd_power_method_tall(A=A, 
                                             AHA=AHA,
                                             inp_dims=self.trj_size,
                                             rank=self.nseg,
                                             device=self.torch_dev)
            spatial_funcs = rearrange(U * S, '... nseg -> nseg ...')
            temporal_funcs = rearrange(Vh, 'nseg ... -> ... nseg')

        else:
            # Build spatio temporal matrix
            A = self.phase_est_slice(slice(None), slice(None), lowrank=False)
            if self.verbose:
                print(f'Calculating SVD')
            A = A.reshape((-1, *self.trj_size))
            A = A.reshape((A.shape[0], -1))
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            spatial_funcs = (S[:self.nseg] * U[:, :self.nseg]).reshape((*self.im_size, self.nseg))
            spatial_funcs = rearrange(spatial_funcs, '... nseg -> nseg ...')
            temporal_funcs = Vh[:self.nseg, :].T.reshape((*self.trj_size, self.nseg))        

        return spatial_funcs, temporal_funcs

    def _calc_temporal_interpolators(self,
                                     betas: torch.Tensor,
                                     interp_type: Optional[str] = 'zero') -> torch.Tensor:
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
        interp_type : str
            interpolator type from the list
                'zero' - zero order interpolator
                'linear' - linear interpolator 
                'lstsq' - least squares interpolator
        
        Returns:
        --------
        interp_coeffs : torch.tensor <complex64>
            interpolation coefficients 'h_l(t)' with shape (*trj_size, nseg)
        """

        # alphas -> (*trj_size, nbasis)
        # betas  -> (nseg,       nbasis)
        # hl(t)  -> (*trj_size, nseg)

        # Consts
        assert self.torch_dev == betas.device
        nseg = betas.shape[0]
        trj_size = self.alphas.shape[:-1]

        # Least squares interpolator
        if 'lstsq' in interp_type:

            # TODO OptimizeME with quantization/histograms

            # Prep AHA, AHB matrices
            alphas_flt = rearrange(self.alphas, '... nbasis -> (...) nbasis').type(torch.float32)
            phis_flt = rearrange(self.phis, 'nbasis ... -> (...) nbasis').type(torch.float32)
            AHA = torch.zeros((nseg, nseg), 
                              dtype=torch.complex64, device=self.torch_dev)
            AHb = torch.zeros((nseg, alphas_flt.shape[0]), 
                              dtype=torch.complex64, device=self.torch_dev)
            
            # Compute AHA and AHB in batches
            batch_size = self.spatial_batch_size
            for n1 in tqdm(range(0, phis_flt.shape[0], batch_size), 'Least Squares Interpolators'):
                n2 = min(n1 + batch_size, phis_flt.shape[0])

                # Accumulate AHA
                A_batch = einsum(phis_flt[n1:n2, :], betas, 
                                 'B nbasis, nseg nbasis -> B nseg')
                A_batch = torch.exp(-2j * torch.pi * A_batch)
                AHA += A_batch.H @ A_batch / (n2 - n1)

                # Accumulate AHb
                B_batch = einsum(phis_flt[n1:n2, :], alphas_flt,
                                 'B nbasis, T nbasis -> B T')
                B_batch = torch.exp(-2j * torch.pi * B_batch)
                AHb += A_batch.H @ B_batch / (n2 - n1)

            # Solve for x = (AHA)^{-1} AHb
            x = lin_solve(AHA, AHb, solver='pinv')
            
            # Reshape (nseg, T)
            interp_coeffs = x.T.reshape((*trj_size, nseg))
            
        # Linear interpolator
        elif 'lin' in interp_type:
            raise NotImplementedError
        
        # Zero order hold/nearest interpolator
        else:
        
            # Empty return coefficients
            interp_coeffs = torch.zeros((*trj_size, nseg), dtype=torch.float32, device=self.torch_dev)

            # Find closest points
            tup = (slice(None),) + (None,) * (self.alphas.ndim - 1) + (slice(None),)
            inds = torch.argmin(
                torch.linalg.norm(self.alphas[None, ...] - betas[tup], dim=-1),
                dim=0) # *trj_size -> values in [0, ..., nseg-1]
            
            # Indicator function
            for i in range(nseg):
                interp_coeffs[..., i] = 1.0 * (inds == i)
            
            del inds

        return interp_coeffs.type(torch.complex64)
  
    def _calc_spatial_interpolators(self,
                                    thetas: torch.Tensor,
                                    interp_type: Optional[str] = 'zero') -> torch.Tensor:
        """
        Calculates spatial interpolation coefficients b_l(r) 
        from the following model:
        
        e^{-j 2pi phi(r, t)} = sum_l b_l(r) e^{-j 2pi phi_l(t) }

        where 

        phi_l(t) = sum_i thetas_l * alphas_i(t),

        Parameters:
        -----------
        thetas : torch.Tensor <float32>
            what most refer to as frequency bins in MFI, shape is (nseg, nbasis)
        interp_type : str
            interpolator type from the list
                'zero' - zero order interpolator
                'linear' - linear interpolator 
                'lstsq' - least squares interpolator
        
        Returns:
        --------
        interp_coeffs : torch.tensor <float32>
            spatial interpolation coefficients 'b_l(r)' with shape (nseg, *im_size)
        """

        # alphas -> (*trj_size, nbasis)
        # thetas -> (nseg,       nbasis)
        # bl(r)  -> (nseg,     *im_size)

        # Consts
        assert self.torch_dev == thetas.device
        nseg = thetas.shape[0]
        im_size = self.im_size
    
        # Least squares interpolator
        if 'lstsq' in interp_type:

            # Reshape
            alphas_flt = rearrange(self.alphas, '... nbasis -> (...) nbasis').type(torch.float32)
            phis_flt = rearrange(self.phis, 'nbasis ... -> (...) nbasis').type(torch.float32)
            x = torch.zeros((nseg, phis_flt.shape[0]), dtype=torch.complex64, device=self.torch_dev)
            
            # Compute AHA matrix
            A = einsum(thetas, alphas_flt, 
                                'nseg nbasis, T nbasis -> T nseg')
            A = torch.exp(-2j * torch.pi * A)
            AHA = A.H @ A # (nseg, nseg)

            # Spatial batching
            batch_size = self.spatial_batch_size
            for n1 in tqdm(range(0, phis_flt.shape[0], batch_size), 'Least Squares Interpolators', disable=not self.verbose):
                n2 = min(n1 + batch_size, phis_flt.shape[0])

                # Compute B matrix
                B_batch = einsum(phis_flt[n1:n2, :], alphas_flt,
                                 'B nbasis, T nbasis -> T B')
                B_batch = torch.exp(-2j * torch.pi * B_batch)

                # Solve for x = (AHA)^{-1} AHb
                x[:, n1:n2] = lin_solve(AHA, A.H @ B_batch)

            # reshape
            interp_coeffs = x.T.reshape((*im_size, nseg)).moveaxis(-1, 0)

        # Linear interpolator
        elif 'lin' in interp_type:
            raise NotImplementedError
        
        # Zero order hold/nearest interpolator
        else:
            
            # Soln with zeros, fill in later
            interp_coeffs = torch.zeros((nseg, *self.im_size), dtype=torch.complex64, device=self.torch_dev)

            # Find closest points
            tup =  (slice(None),) + (None,) * (len(im_size)) + (slice(None),)
            inds = torch.argmin(
                torch.linalg.norm(self.phis[..., None] - thetas.T[tup], dim=0),
                dim=-1) # *im_size -> values in [0, ..., nseg-1]
            
            # Indicator function
            for i in range(nseg):
                interp_coeffs[i, ...] = 1.0 * (inds == i)

        return interp_coeffs

    # -------- Plotting Functions --------

    def _plot_approx_err(self,
                         t_slice):
        
        import matplotlib.pyplot as plt

        assert len(self.im_size) == 2

        r_slice = (slice(None),) * 2
        phase_est = self.phase_est_slice(r_slice, t_slice, lowrank=True)
        phase = self.phase_est_slice(r_slice, t_slice, lowrank=False)

        if phase.ndim == 2:
            phase = phase[..., None]
            phase_est = phase_est[..., None]
        true_phase = torch.angle(phase).cpu().mean(dim=-1).cpu()
        est_phase = torch.angle(phase_est).cpu().mean(dim=-1).cpu()
        err = torch.mean(torch.abs(phase_est - phase), dim=-1).cpu()

        plt.figure(figsize=(14,7))
        if 'svd' in self.method:
            plt.suptitle(f'{self.method.upper()} Model, {self.nseg} Segments')
        else:
            plt.suptitle(f'{self.method.upper()} Model, {self.interp_type.upper()} Interpolator, {self.nseg} Segments')
        plt.subplot(131)
        plt.title('True Phase')
        plt.imshow(torch.rad2deg(true_phase), vmin=-180, vmax=180, cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(132)
        plt.title('Estimated Phase')
        plt.imshow(torch.rad2deg(est_phase), vmin=-180, vmax=180, cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(133)
        plt.title('Error = |True - Estimated|')
        plt.imshow(err, cmap='gray', vmin=0, vmax=0.5)
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()

    def _phase_movie(self, t_slice, r_slice=None, name='test.gif'):

        
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1,2)

        ims = []

        ndim = len(self.im_size)
        if r_slice is None:
            if ndim == 3:            
                slc = self.im_size[-1]//2
                r_slice = (slice(None),) * (ndim - 1) + (slc,)
            else:
                r_slice = (slice(None),) * ndim
        phase = self.phase_est_slice(r_slice, t_slice, lowrank=False).cpu()
        phase_LR = self.phase_est_slice(r_slice, t_slice, lowrank=True).cpu()
        vmin = torch.angle(phase).min()
        vmax = torch.angle(phase).max()

        ax1.set_title('True Phase')
        ax2.set_title('Time-Segmented Phase Model')
        ax1.axis('off')
        ax2.axis('off')
        for i in tqdm(range(phase.shape[-1]), 'Making Movie'):
            im1 = ax1.imshow(torch.angle(phase[..., i]), vmin=vmin, vmax=vmax, animated=True, cmap='jet')
            im2 = ax2.imshow(torch.angle(phase_LR[..., i]), vmin=vmin, vmax=vmax, cmap='jet')
            ims.append([im1,im2])
        width = 0.75
        cb_ax = fig.add_axes([(1-width)/2,.13,width,.04])
        fig.colorbar(im2,orientation='horizontal',cax=cb_ax)
        fig.tight_layout()
        ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                repeat_delay=500)
        writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
        ani.save(name, writer=writer)
        quit()
