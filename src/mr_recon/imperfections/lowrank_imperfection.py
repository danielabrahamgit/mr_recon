import torch

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
from mr_recon.fourier import torchkb_nufft, sigpy_nufft
from mr_recon.algs import lin_solve, svd_power_method_tall

class lowdim_imperfection:
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

        if self.method not in ['ts', 'svd']:
            raise ValueError('method must be one of ["ts", "svd"]')
        
        if self.interp_type not in ['zero', 'linear', 'lstsq']:
            raise ValueError('interp_type must be one of ["zero", "linear", "lstsq"]')
        
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
    
class exponential_imperfection(lowdim_imperfection):
    """
    Broadly represents class of exponential imperfections. These will have the form:
    b(t) = int_r m(r) T(t){s(r)} e^{-j 2pi k(t) r} dr
    
    Where T(t){s(r)} = s(r) e^{-j 2pi phi(r) @ alpha(t)}

    where phi(r) are the spatial maps and alpha(t) are the temporal terms.
    There are B phis and B alphas
    """

    def __init__(self, 
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 L: int,
                 method: Optional[str] = 'ts',
                 interp_type: Optional[str] = 'zero',
                 verbose: Optional[bool] = True):
        """
        Parameters:
        -----------
        phis : torch.Tensor
            The spatial maps with shape (B, *im_size)
        alphas : torch.Tensor
            The temporal maps with shape (B, *trj_size)
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
        self.alphas = alphas
        self.phis = phis
        self.B = phis.shape[0]
        self.im_size = phis.shape[1:]
        self.trj_size = alphas.shape[1:]
        self.torch_dev = phis.device

        assert alphas.device == self.torch_dev, 'alphas and phis must be on same device'
        assert alphas.dtype == torch.float32, 'alphas must be float32'
        assert phis.dtype == torch.float32, 'phis must be float32'

        super().__init__(L, method, interp_type, verbose)

    def _calc_time_segmentation(self,
                                spatial_batch_size: Optional[int] = 2 ** 8) -> None:
        """
        Computes both h_l and T_l functions using time segmentation.

        T_l{x(r)} = e^{- phi(r) @ self.alpha_clusters[l]} x(r)
        h_l(t) = self.interp_funcs(l, t)

        Parameters:
        -----------
        spatial_batch_size : int
            batch size over spatial terms

        Saves:
        ------
        self.alpha_clusters : torch.Tensor
            The clusters of alphas with shape (L, B)
        self.interp_funcs : torch.Tensor
            The interpolating functions with shape (L, *trj_size)
        """

        # If 1D, do uniform
        if self.B == 1:
            method = 'uniform'
        # Otherwise, use k-means
        else:
            method = 'cluster'
        alpha_clusters, idxs = quantize_data(self.alphas.reshape((self.B, -1)).T, 
                                             self.L, method=method)
        self.alpha_clusters = alpha_clusters
        idxs = idxs.reshape(self.trj_size)
            
        if 'lstsq' in self.interp_type:

            # Prep AHA, AHB matrices
            alphas_flt = rearrange(self.alphas, 'nbasis ... -> (...) nbasis')
            phis_flt = rearrange(self.phis, 'nbasis ... -> (...) nbasis')
            AHA = torch.zeros((self.L, self.L), 
                              dtype=self.phis.dtype, device=self.torch_dev)
            AHb = torch.zeros((self.L, alphas_flt.shape[0]), 
                              dtype=self.phis.dtype, device=self.torch_dev)
            
            # Compute AHA and AHB in batches
            batch_size = spatial_batch_size
            for n1 in tqdm(range(0, phis_flt.shape[0], batch_size), 'Least Squares Interpolators'):
                n2 = min(n1 + batch_size, phis_flt.shape[0])

                # Accumulate AHA
                A_batch = einsum(phis_flt[n1:n2, :], self.alpha_clusters, 
                                 'B nbasis, L nbasis -> B L')
                A_batch = torch.exp(-2j * torch.pi * A_batch)
                AHA += A_batch.H @ A_batch / (n2 - n1)

                # Accumulate AHb
                B_batch = einsum(phis_flt[n1:n2, :], alphas_flt,
                                 'B nbasis, T nbasis -> B T')
                B_batch = torch.exp(-2j * torch.pi * B_batch)
                AHb += A_batch.H @ B_batch / (n2 - n1)

            # Solve for x = (AHA)^{-1} AHb
            x = lin_solve(AHA, AHb, solver='pinv')
            
            # Reshape (L, T)
            interp_funcs = x.reshape((self.L, *self.trj_size))
        elif 'linear' in self.interp_type:
            raise NotImplementedError
        elif 'zero' in self.interp_type:
            # Indicator function
            interp_funcs = torch.zeros((self.L, *self.trj_size,), dtype=self.phis.dtype, device=self.torch_dev)
            for i in range(self.L):
                interp_funcs[i, ...] = 1.0 * (idxs == i)
        else:
            raise ValueError('interp_type must be one of ["zero", "linear", "lstsq"]')

        self.temporal_funcs = interp_funcs

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

        return self._calc_svd_naive()
       
    def _calc_svd_naive(self) -> torch.Tensor:
        """
        Calculates spatial and temporal functions from SVD 
        of spatio-temporal matrix. This probably won't work for 
        larger problems, use the batched version istead for those.
        
        Returns:
        --------
        spatial_funcs : torch.tensor
            spatial basis functions with shape (nseg, *im_size)
        temporal_funcs : torch.tensor
            temporal basis functions with shape (*trj_size, nseg)
        """
        
        # Build spatio temporal matrix
        if self.verbose:
            print(f'Calculating SVD')
        A = self._build_spatio_temporal_matrix(flatten=True, lowrank=False)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        spatial_funcs = (S[:self.L] * U[:, :self.L]).reshape((*self.im_size, self.L))
        spatial_funcs = rearrange(spatial_funcs, '... nseg -> nseg ...')
        temporal_funcs = Vh[:self.L, :].reshape(( self.L, *self.trj_size))        

        return spatial_funcs, temporal_funcs

    def _calc_svd_batched(self,
                          spatial_batch_size: Optional[int] = 2 ** 14) -> torch.Tensor:
        """
        Calculates spatial and temporal functions from SVD in a batched 
        operator way.

        Parameters:
        -----------
        spatial_batch_size : int
            batch size over spatial terms
        
        Returns:
        --------
        spatial_funcs : torch.tensor 
            spatial basis functions with shape (nseg, *im_size)
        temporal_funcs : torch.tensor 
            temporal basis functions with shape (*trj_size, nseg)
        """

        def A(x):
            x = x.reshape((-1))
            nvox = torch.prod(torch.tensor(self.im_size)).item()
            y = torch.zeros((nvox,), dtype=torch.complex64, device=x.device)
            for n1, n2 in batch_iterator(nvox, spatial_batch_size):
                y[n1:n2] = self._build_spatio_temporal_matrix(r_slice=slice(n1, n2), 
                                                              t_slice=slice(None), 
                                                              flatten=True,
                                                              lowrank=False) @ x
            return y.reshape(self.im_size)
        
        def AH(y):
            y = y.reshape((-1))
            nvox = torch.prod(torch.tensor(self.im_size)).item()
            ntrj = torch.prod(torch.tensor(self.trj_size)).item()
            x = torch.zeros((ntrj,), dtype=torch.complex64, device=y.device)
            for n1, n2 in batch_iterator(nvox, spatial_batch_size):
                x += self._build_spatio_temporal_matrix(r_slice=slice(n1, n2), 
                                                        t_slice=slice(None), 
                                                        flatten=True,
                                                        lowrank=False).H @ y[n1:n2]
            return x.reshape(self.trj_size)
        
        def AHA(x):
            return AH(A(x))
            
        U, S, Vh = svd_power_method_tall(A=A, 
                                         AHA=AHA,
                                         inp_dims=self.trj_size,
                                         rank=self.L,
                                         device=self.torch_dev)
        spatial_funcs = rearrange(U * S, '... nseg -> nseg ...')
        temporal_funcs = Vh

        return spatial_funcs, temporal_funcs
    
    def _calc_svd_fourier_time(self) -> torch.Tensor:
        """
        Calculates spatial and temporal functions from SVD using the 
        fourier and Toeplitz method in time.

        ** Assumes that alpha(t) = t **
        where t starts at zero
        TODO - make this more general by letting t start anytime

        Parameters:
        -----------
        spatial_batch_size : int
            batch size over spatial terms
        
        Returns:
        --------
        spatial_funcs : torch.tensor 
            spatial basis functions with shape (nseg, *im_size)
        temporal_funcs : torch.tensor 
            temporal basis functions with shape (*trj_size, nseg)
        """

        # Make sure this is the right thing to do
        assert self.B == 1, 'This method only works for B=1'
        alphas_flt = self.alphas.squeeze()
        assert alphas_flt.ndim == 1, 'This method only works for 1D broadcastable alphas'
        nro = alphas_flt.shape[0]
        
        # Build operators via fourier method
        nufft = torchkb_nufft(im_size=(nro,), device_idx=self.torch_dev.index)
        freqs = nufft.rescale_trajectory(self.phis * nro)[0, ..., None]
        x_0 = torch.zeros((nro,), dtype=torch.complex64, device=self.torch_dev)
        x_0[0] = 1
        phase_0 = nufft(x_0[None,], freqs[None,])[0]
        def A(x):
            return nufft(x[None], freqs[None])[0] * phase_0.conj() * nro
        # def AH(y):
        #     return nufft.adjoint(y[None,] * phase_0, freqs[None])[0] * nro
        kerns = nufft.calc_teoplitz_kernels(freqs[None])
        def AHA(x):
            return nufft.normal_toeplitz(x[None,None,], kerns)[0,0]
        
        # Compute SVD
        U, S, Vh = svd_power_method_tall(A=A, 
                                         AHA=AHA,
                                         inp_dims=(nro,),
                                         rank=self.L,
                                         device=self.torch_dev)
        spatial_funcs = rearrange(U * S, '... nseg -> nseg ...')
        temporal_funcs = Vh.reshape((self.L, *self.trj_size))

        return spatial_funcs, temporal_funcs
                                         
    def _calc_svd_fourier_space(self,
                                spatial_batch_size: Optional[int] = 2 ** 14) -> torch.Tensor:
        """
        Calculates spatial and temporal functions from SVD using the 
        Fourier and Toeplitz method in space.

        ** Assumes that phi(r) = r **
        ** Make sure r is from -1/2 to 1/2 **
        ** Make sure alpha(t) is from -N/2, N/2 in each axis **

        Parameters:
        -----------
        spatial_batch_size : int
            batch size over spatial terms
        
        Returns:
        --------
        spatial_funcs : torch.tensor 
            spatial basis functions with shape (nseg, *im_size)
        temporal_funcs : torch.tensor 
            temporal basis functions with shape (*trj_size, nseg)
        """
        assert self.B == len(self.im_size)

        # Build operators via fourier method
        nufft = torchkb_nufft(im_size=self.im_size, device_idx=self.torch_dev.index)
        trj = -rearrange(self.alphas, 'B ... -> ... B')
        trj = nufft.rescale_trajectory(trj)
        nvox = torch.prod(torch.tensor(self.im_size)).item()
        def A(x):
            return nufft.adjoint(x[None,], trj[None,])[0] * nvox ** 0.5
        def AH(y):
            return nufft(y[None,], trj[None,])[0] * nvox ** 0.5
        kerns = nufft.calc_teoplitz_kernels(trj[None])
        def AAH(y):
            return nufft.normal_toeplitz(y[None,None,], kerns)[0,0] * nvox
        
        # Compute SVD
        U, S, Vh = svd_power_method_tall(A=AH,
                                         AHA=AAH,
                                         niter=100,
                                         inp_dims=self.im_size,
                                         rank=self.L,
                                         device=self.torch_dev)
        self.temporal_funcs = rearrange(U * S, '... nseg -> nseg ...').conj()
        self.spatial_funcs = Vh.conj()
        
        # ro, pe = 300, 0
        # kern_test = self._build_spatio_temporal_matrix(t_slice=(ro, pe), flatten=False, lowrank=False)
        # kern_svd = self._build_spatio_temporal_matrix(t_slice=(ro, pe), flatten=False, lowrank=True)
        # # kern_svd = self.spatial_funcs[0].abs() + 1j * self.spatial_funcs[1].abs()
        # # sf, tf = self._calc_svd_batched()
        # # kern_test = sf[0].abs() + 1j * sf[1].abs()
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(14,7))
        # plt.subplot(221)
        # plt.title('True Kernel')
        # plt.imshow(kern_test.real.cpu())
        # plt.axis('off')
        # plt.subplot(222)
        # plt.title('SVD Kernel')
        # plt.imshow(kern_svd.real.cpu())
        # plt.axis('off')
        # plt.subplot(223)
        # plt.imshow(kern_test.imag.cpu())
        # plt.axis('off')
        # plt.subplot(224)
        # plt.imshow(kern_svd.imag.cpu())
        # plt.axis('off')
        # plt.tight_layout()
        # plt.figure(figsize=(10,10))
        # plt.imshow((kern_test - kern_svd).abs().cpu())
        # plt.axis('off')
        # plt.show()
        # quit()

        return self.spatial_funcs, self.temporal_funcs

    def apply_spatial(self, 
                      x: Optional[torch.Tensor] = None, 
                      ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        if self.method == 'ts':
            exp_term = einsum(self.phis, self.alpha_clusters[ls], 'B ..., L B -> L ...')
            bs = torch.exp(-2j * torch.pi * exp_term)
        elif self.method == 'svd':
            bs = self.spatial_funcs[ls]
        if x is None:
            return bs
        else:
            return bs * x.unsqueeze(-len(self.im_size)-1)
    
    def apply_spatial_adjoint(self, 
                              y: torch.Tensor, 
                              ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        if self.method == 'ts':
            exp_term = einsum(self.phis, self.alpha_clusters[ls], 'B ..., L B -> L ...')
            bs = torch.exp(-2j * torch.pi * exp_term)
        elif self.method == 'svd':
            bs = self.spatial_funcs[ls]
        return torch.sum(bs.conj() * y, dim=-len(self.im_size)-1)
        
    def apply_temporal(self, 
                       x: torch.Tensor, 
                       ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        h = self.temporal_funcs[ls]
        return (h * x).sum(dim=-len(self.trj_size)-1)

    def apply_temporal_adjoint(self, 
                               y: Optional[torch.Tensor] = None, 
                               ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        h = self.temporal_funcs[ls]
        if y is None:
            return h
        else:
            return h.conj() * y.unsqueeze(-len(self.trj_size)-1)

    def _build_spatio_temporal_matrix(self,
                                      r_slice: Optional[tuple] = slice(None), 
                                      t_slice: Optional[tuple] = slice(None),
                                      flatten: Optional[bool] = False,
                                      lowrank: Optional[bool] = False) -> torch.Tensor:
        """
        Builds the spatio-temporal matrix representing the phase accrual at all 
        time points.

        Parameters:
        -----------
        r_slice : Optional[tuple]
            Slices the spatial terms
        t_slice : Optional[tuple]
            Slices the temporal terms
        flatten : Optional[bool]
            if True, flattens the spatial and temporal dims
        lowrank : Optional[bool]
            if True, uses lowrank approximation
        
        Returns:
        --------
        st_matrix : torch.Tensor
            The spatio-temporal matrix with shape (*im_size(r_slice), *trj_size(t_slice))
        """

        if lowrank:
            st_matrix = None
            for i in range(self.L):
                
                # Get spatial temporal basis funcs
                b = self.apply_spatial(None, ls=i)
                h = self.apply_temporal_adjoint(None, ls=i)
                if flatten:
                    b = b.flatten()[r_slice]
                    h = h.flatten()[t_slice]
                else:
                    b = b[r_slice]
                    h = h[t_slice]
                tup_b = (slice(None),) * b.ndim + (None,) * h.ndim
                tup_h = (None,) * b.ndim + (slice(None),) * h.ndim

                # accumulate
                if st_matrix is None:
                    st_matrix = b[tup_b] * h[tup_h]
                else:
                    st_matrix += b[tup_b] * h[tup_h]
        else:
            st_matrix = None
            for i in range(self.B):

                # Get spatial temporal basis funcs 
                phi = self.phis[i]
                alpha = self.alphas[i]
                if flatten:
                    phi = phi.flatten()[r_slice]
                    alpha = alpha.flatten()[t_slice]
                else:
                    phi = phi[r_slice]
                    alpha = alpha[t_slice]
                tup_p = (slice(None),) * phi.ndim + (None,) * alpha.ndim
                tup_a = (None,) * phi.ndim + (slice(None),) * alpha.ndim

                # accumulate
                if st_matrix is None:
                    st_matrix = phi[tup_p] * alpha[tup_a]
                else:
                    st_matrix += phi[tup_p] * alpha[tup_a]
            
            st_matrix = torch.exp(-2j * torch.pi * st_matrix)

        return st_matrix 