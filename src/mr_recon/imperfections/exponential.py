import torch

from tqdm import tqdm
from typing import Optional
from einops import rearrange, einsum
from mr_recon.fourier import sigpy_nufft
from mr_recon.algs import lin_solve, svd_power_method_tall
from mr_recon.imperfections.imperfection import imperfection
from mr_recon.utils import (
    quantize_data,
    batch_iterator
)

class exponential_imperfection(imperfection):
    """
    Broadly represents class of complex exponential imperfections. These will have the form:
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
                 verbose: Optional[bool] = True,
                 complex_exp: Optional[bool] = True):
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
        complex_exp : bool
            if True, uses complex exponential model e^{-j2pi alphas * phis}
            if False uses real exponential model e^{-alphas * phis}
        """
        self.alphas = alphas
        self.phis = phis
        self.B = phis.shape[0]
        self.im_size = phis.shape[1:]
        self.trj_size = alphas.shape[1:]
        self.torch_dev = phis.device
        if complex_exp:
            self.exp_scale = 2j * torch.pi
        else:
            self.exp_scale = 1

        assert alphas.device == self.torch_dev, 'alphas and phis must be on same device'
        assert alphas.dtype == torch.float32, 'alphas must be float32'
        assert phis.dtype == torch.float32, 'phis must be float32'

        super().__init__(L, method, interp_type, verbose)

    def _calc_time_segmentation(self,
                                spatial_batch_size: Optional[int] = None,
                                temporal_batch_size: Optional[int] = 2 ** 8,
                                sketch_AHb: Optional[bool] = False) -> None:
        """
        Computes both h_l and T_l functions using time segmentation.

        T_l{x(r)} = e^{- phi(r) @ self.alpha_clusters[l]} x(r)
        h_l(t) = self.interp_funcs(l, t)

        Parameters:
        -----------
        spatial_batch_size : int
            batch size over spatial terms
        temporal_batch_size : int
            batch size over temporal terms

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

            # Flatten spatial and temporal terms
            alphas_flt = rearrange(self.alphas, 'nbasis ... -> (...) nbasis')
            phis_flt = rearrange(self.phis, 'nbasis ... -> (...) nbasis')
            T = alphas_flt.shape[0]
            N = phis_flt.shape[0]

            # Defualt batch
            if spatial_batch_size is None:
                spatial_batch_size = N
            if temporal_batch_size is None:
                temporal_batch_size = T

            # Desired temporal functions
            interp_funcs = torch.zeros((self.L, T),
                                        dtype=torch.complex64, device=self.torch_dev)
            
            # Compute AHA
            AHA = torch.zeros((self.L, self.L), 
                              dtype=torch.complex64, device=self.torch_dev)
            for n1, n2 in batch_iterator(N, spatial_batch_size):
                n2 = min(n1 + spatial_batch_size, N)
                A_batch = phis_flt[n1:n2] @ self.alpha_clusters.T
                A_batch = torch.exp(-self.exp_scale * A_batch)
                AHA += A_batch.H @ A_batch / N
            # AHA_inv = torch.linalg.inv(AHA)

            # Heavy batching for AHb
            for t1 in tqdm(range(0, T, temporal_batch_size), 'Least Squares Interpolators'):
                t2 = min(t1 + temporal_batch_size, T)

                # Compute AHb
                AHb = torch.zeros((self.L, t2 - t1), 
                                dtype=torch.complex64, device=self.torch_dev)

                # Sketch over voxel dimension
                if sketch_AHb:
                    pass

                # Combute AHb by batching over all voxels - this is slow!
                else:
                    for n1, n2 in batch_iterator(N, spatial_batch_size):
                        n2 = min(n1 + spatial_batch_size, N)
                        A_batch = phis_flt[n1:n2] @ self.alpha_clusters.T
                        A_batch = torch.exp(-self.exp_scale * A_batch)
                        B_batch = phis_flt[n1:n2, :] @ alphas_flt[t1:t2].T
                        B_batch = torch.exp(-self.exp_scale * B_batch)
                        AHb += A_batch.H @ B_batch / N

                # Solve for ls = (AHA)^{-1} AHb
                ls = lin_solve(AHA, AHb, solver='pinv')
                # ls = AHA_inv @ AHb
                interp_funcs[:, t1:t2] = ls
            
            # Reshape (L, T)
            interp_funcs = interp_funcs.reshape((self.L, *self.trj_size))
        elif 'linear' in self.interp_type:
            raise NotImplementedError
        elif 'zero' in self.interp_type:
            # Indicator function
            interp_funcs = torch.zeros((self.L, *self.trj_size,), dtype=torch.complex64, device=self.torch_dev)
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
    
    def _calc_svd_fourier_time(self,
                               use_topeplitz: Optional[bool] = True) -> torch.Tensor:
        """
        Calculates spatial and temporal functions from SVD using the 
        fourier and Toeplitz method in time.

        ** Assumes that alpha(t) = t **
        where t starts at zero
        TODO - make this more general by letting t start anytime

        Parameters:
        -----------
        use_topeplitz : bool
            if True, uses toeplitz for extra speed
        
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
        phis = self.phis
        assert alphas_flt.ndim == 1, 'This method only works for 1D broadcastable alphas'
        nro = alphas_flt.shape[0]
        assert nro % 2 == 0
        nvox = torch.prod(torch.tensor(self.im_size)).item()
        
        # Use NUFFT to evaluate A : int_t x(t) e^{-j 2pi t phi(r)} dt
        # And its adjoint/normal operator
        nufft = sigpy_nufft(im_size=(nro,))
        freqs = nufft.rescale_trajectory(phis * nro)[0, ..., None]

        # Compute phase_0 to make sure that no phase is applied at first time point
        x_0 = torch.zeros((nro,), dtype=torch.complex64, device=self.torch_dev)
        x_0[0] = 1
        phase_0 = nufft(x_0[None,], freqs[None,])[0] * nro / nvox # scaling nro / nvox helps with numerical stability

        # Define forward and adjoint operators
        def A(x):
            return nufft(x[None], freqs[None])[0] * phase_0.conj()
        def AH(y):
            return nufft.adjoint(y[None,] * phase_0, freqs[None])[0]
        
        # Use toeplitz for extra speed
        if use_topeplitz:
            weights = phase_0 * phase_0.conj()
            kerns = nufft.calc_teoplitz_kernels(freqs[None,], weights[None,])
            def AHA(x):
                return nufft.normal_toeplitz(x[None,None,], kerns)[0,0]
        else:
            def AHA(x):
                return AH(A(x))
        
        # Compute SVD
        U, S, Vh = svd_power_method_tall(A=A, 
                                         AHA=AHA,
                                         inp_dims=(nro,),
                                         rank=self.L,
                                         device=self.torch_dev)
        spatial_funcs = rearrange(U * S, '... nseg -> nseg ...') * nvox
        temporal_funcs = Vh.reshape((self.L, *self.trj_size))

        return spatial_funcs, temporal_funcs
                                         
    def _calc_svd_fourier_space(self,
                                use_topeplitz: Optional[bool] = True) -> torch.Tensor:
        """
        Calculates spatial and temporal functions from SVD using the 
        Fourier and Toeplitz method in space.

        ** Assumes that phi(r) = r **
        ** Make sure r is from -1/2 to 1/2 **
        ** Make sure alpha(t) is from -N/2, N/2 in each axis **

        Parameters:
        -----------
        use_topeplitz : bool
            if True, uses toeplitz for extra speed
        
        Returns:
        --------
        spatial_funcs : torch.tensor 
            spatial basis functions with shape (nseg, *im_size)
        temporal_funcs : torch.tensor 
            temporal basis functions with shape (*trj_size, nseg)
        """
        assert self.B == len(self.im_size)

        # Build operators via fourier method
        nufft = sigpy_nufft(im_size=self.im_size)
        trj = -rearrange(self.alphas, 'B ... -> ... B')
        trj = nufft.rescale_trajectory(trj)
        nvox = torch.prod(torch.tensor(self.im_size)).item()
        def A(x):
            return nufft.adjoint(x[None,], trj[None,])[0] * nvox ** 0.5
        def AH(y):
            return nufft(y[None,], trj[None,])[0] * nvox ** 0.5
        kerns = nufft.calc_teoplitz_kernels(trj[None])

        if use_topeplitz:
            def AAH(y):
                return nufft.normal_toeplitz(y[None,None,], kerns)[0,0] * nvox
        else:
            def AAH(y):
                return A(AH(y))
        
        # Compute SVD
        U, S, Vh = svd_power_method_tall(A=AH,
                                         AHA=AAH,
                                         niter=100,
                                         inp_dims=self.im_size,
                                         rank=self.L,
                                         device=self.torch_dev)
        self.temporal_funcs = rearrange(U * S, '... nseg -> nseg ...').conj()
        self.spatial_funcs = Vh.conj()
    
        return self.spatial_funcs, self.temporal_funcs

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
            The spatio-temporal matrix with shape (... *im_size(r_slice), *trj_size(t_slice))
        """

        if lowrank:
            st_matrix = None
            for i in range(self.L):
                
                # Get spatial temporal basis funcs
                b = self.apply_spatial(None, ls=slice(i, i+1))[0]
                h = self.apply_temporal_adjoint(None, ls=slice(i, i+1))[0]
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
            
            st_matrix = torch.exp(-self.exp_scale * st_matrix)

        return st_matrix 
    
    def get_network_features(self) -> torch.Tensor:
        """
        Relevant network features are the alpha coefficients and the hl functions

        Returns:
        --------
        features : torch.Tensor
            The features of the imperfection with shape (*trj_size, nfeat)
        """
        if self.method == 'ts' and self.interp_type == 'zero':
            self.temporal_funcs /= self.temporal_funcs.abs().max()
            features = torch.cat((self.alphas, 
                                self.temporal_funcs.real, 
                                self.temporal_funcs.imag), dim=0)
        else:
            features = self.alphas
        features = rearrange(features, 'nfeat ... -> ... nfeat')
        return features
    
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
        
        # Flatten spatial and temporal dims
        x = x.flatten(start_dim=-len(self.im_size))[..., r_inds]

        if lowrank:
            h = self.apply_temporal_adjoint().reshape((self.L, -1))[:, t_inds]
            b = self.apply_spatial().reshape((self.L, -1))[:, r_inds]
            xt = torch.sum(b * h, dim=0) * x
        else:
            alphas = self.alphas.reshape((self.B, -1))[:, t_inds]
            phis = self.phis.reshape((self.B, -1))[:, r_inds]
            xt = torch.exp(-self.exp_scale * torch.sum(alphas * phis, dim=0)) * x
        return xt
    
    def apply_spatial(self, 
                      x: Optional[torch.Tensor] = None, 
                      ls: Optional[torch.Tensor] = slice(None)) -> torch.Tensor:
        if self.method == 'ts':
            exp_term = einsum(self.phis, self.alpha_clusters[ls], 'B ..., L B -> L ...')
            bs = torch.exp(-self.exp_scale * exp_term)
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
            bs = torch.exp(-self.exp_scale * exp_term)
        elif self.method == 'svd':
            bs = self.spatial_funcs[ls]
        return (bs.conj() * y).sum(dim=-len(self.im_size)-1)
        
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
