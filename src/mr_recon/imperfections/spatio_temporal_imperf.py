import torch

from typing import Optional, Union
from einops import rearrange, einsum
from mr_recon.dtypes import complex_dtype
from mr_recon.pad import PadLast
from mr_recon.fourier import sigpy_nufft, fft, ifft
from mr_recon.utils import gen_grd, quantize_data, batch_iterator
from mr_recon.spatial import spatial_resize
from math import ceil

class spatio_temporal:
    """
    This class represents imperfections that takes the form W(r, p(t)),
    where r is spatial and p(t) are temporal features.

    This means the received signal from the cth coil is given by:
    s_c(t) = int_r m(r) * W(r, p(t)) * S_c(r) * e^{-j 2pi k(t) * r} dr

    Examples:
    ---------
    B0/Off-Resonance:
        W(r, p(t)) = exp(-j 2pi B0(r) * t)
        p(t) = t
    Eddy Currents:
        W(r, p(t)) = exp(-j 2pi sum_k phi_k(r) * alpha_k(t))
        p(t) = [alpha_1(t), alpha_2(t), ..., alpha_K(t)]
    Motion:
        W(r, p(t)) = S(Rotate(theta(t)) * r + u(t)) / S(r)
        p(t) = [theta(t), u(t)]
    """

    def __init__(self,
                 im_size: tuple,
                 trj_size: tuple,
                 spatial_batch_size: Optional[int] = None,
                 temporal_batch_size: Optional[int] = None,
                 device: Optional[torch.device] = torch.device('cpu')):
        """
        Parameters:
        -----------
        im_size: tuple
            the size of the image, should be (N_0, N_1, ..., N_{d-1})
            d is the number of spatial dimensions
        trj_size: tuple
            the size of the trajectory (M_0, M_1, ..., M_{k-1})
            k is the number of temporal dimensions
        """
        self.im_size = im_size
        self.trj_size = trj_size
        self.torch_dev = device
        self.nvox = torch.prod(torch.tensor(im_size)).item()
        self.ntrj = torch.prod(torch.tensor(trj_size)).item()
        self.spatial_batch_size = self.nvox if spatial_batch_size is None else spatial_batch_size
        self.temporal_batch_size = self.ntrj if temporal_batch_size is None else temporal_batch_size

    def temporal_features(self, 
                          temporal_inds: torch.Tensor) -> torch.Tensor:
        """
        Returns the temporal features p(t) at the given indices.

        Parameters:
        -----------
        temporal_inds: torch.Tensor <int>
            the temporal index with shape (k, ...)
        
        Returns:
        --------
        torch.Tensor
            the temporal features p(t) at the given indices with shape (f, ...)
            f is the number of temporal features
            same as p(t)
        """
        raise NotImplementedError

    def get_temporal_clusters(self,
                              num_clusters: int) -> Union[torch.Tensor, torch.Tensor]:
        """
        Returns clusters and indices of temporal features.

        Parameters:
        -----------
        num_clusters: int
            the number of clusters
        
        Returns:
        --------
        clusters: torch.Tensor
            the temporal clusters with shape (f, num_clusters)
        inds: torch.Tensor <int>
            the indices of the temporal clusters with shape (*trj_size), values in [0, num_clusters-1]
        """
        raise NotImplementedError

    def matrix_access(self,
                      spatial_inds: torch.Tensor,
                      temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Returns the value of W(r, p(t)) at the given indices.

        Parameters:
        -----------
        spatial_inds: torch.Tensor <int>
            the spatial indices with shape (d, ...)
        temporal_features: torch.Tensor <int>
            the temporal features with shape (f, ...)
            same as p(t)
        
        Returns:
        --------
        torch.Tensor
            the value of W(r, p(t)) at the given indices with shape (...)
        """
        raise NotImplementedError

    def _continuous_forward_matrix_prod(self,
                                        spatial_input: torch.Tensor,
                                        temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Applies W(r, p(t)) to spatial input vector:

        out(t) = int_r W(r, p(t)) * input(r) dr

        Parameters:
        -----------
        spatial_input: torch.Tensor
            the spatial input vector with shape (N, *im_size)
            N is a batch dimension
        temporal_features: torch.Tensor
            the temporal features with shape (f, ...)
        
        Returns:
        --------
        torch.Tensor
            the output temporal vector with shape (N, ...)
        """
        
        # Consts
        spatial_batch_size = self.spatial_batch_size
        temporal_batch_size = self.temporal_batch_size
        im_size = self.im_size
        d = len(im_size)
        
        # Flatten input
        spatial_input_flt = spatial_input.view(-1, self.nvox)
        N = spatial_input_flt.shape[0]
        
        # Indices for matrix access
        f, *temporal_shape = temporal_features.shape
        temporal_features = temporal_features.view(f, -1)
        spatial_inds = gen_grd(self.im_size, self.im_size) + torch.tensor(self.im_size) // 2
        spatial_inds = spatial_inds.view((-1, len(self.im_size))).T.type(torch.int)
        
        # Batched matrix product
        output = torch.zeros((N, temporal_features.shape[-1]), device=spatial_input_flt.device, dtype=spatial_input_flt.dtype)
        for r1, r2 in batch_iterator(self.nvox, spatial_batch_size):
            for t1, t2 in batch_iterator(temporal_features.shape[-1], temporal_batch_size):
                spatial_inds_batch = torch.zeros((d, r2-r1, t2-t1), dtype=torch.int, device=self.torch_dev)
                temporal_feat_batch = torch.zeros((f, r2-r1, t2-t1), dtype=temporal_features.dtype, device=self.torch_dev)
                spatial_inds_batch[...] = spatial_inds[:, r1:r2, None]
                temporal_feat_batch[...] = temporal_features[:, None, t1:t2]
                mat = self.matrix_access(spatial_inds_batch, temporal_feat_batch)
                output[:, t1:t2] += spatial_input_flt[:, r1:r2] @ mat
                
        return output.view(N, *temporal_shape)

    def _continuous_adjoint_matrix_prod(self,
                                        temporal_input: torch.Tensor,
                                        temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Applies W(r, p(t)) to temporal input vector:

        out(r) = int_t conj(W(r, p(t))) * input(t) dt

        Parameters:
        -----------
        temporal_input
            the input temporal vector with shape (N, ...)
        temporal_features
            the temporal features with shape (f, ...)
        
        Returns:
        --------
        torch.Tensor
            the spatial output vector with shape (N, *im_size)
        """
        
        # Consts
        spatial_batch_size = self.spatial_batch_size
        temporal_batch_size = self.temporal_batch_size
        im_size = self.im_size
        d = len(im_size)
        
        # Indices for matrix access
        f, *temporal_shape = temporal_features.shape
        temporal_features = temporal_features.view(f, -1)
        spatial_inds = gen_grd(self.im_size, self.im_size) + torch.tensor(self.im_size) // 2
        spatial_inds = spatial_inds.view((-1, len(self.im_size))).T.type(torch.int)
        
        # Flatten input
        temporal_input_flt = temporal_input.view((-1, temporal_features.shape[1]))
        N = temporal_input_flt.shape[0]
        
        # Batched matrix product
        output = torch.zeros((N, self.nvox), device=temporal_input_flt.device, dtype=temporal_input_flt.dtype)
        for r1, r2 in batch_iterator(self.nvox, spatial_batch_size):
            for t1, t2 in batch_iterator(temporal_features.shape[-1], temporal_batch_size):
                spatial_inds_batch = torch.zeros((d, t2-t1, r2-r1), dtype=torch.int, device=self.torch_dev)
                temporal_feat_batch = torch.zeros((f, t2-t1, r2-r1), dtype=temporal_features.dtype, device=self.torch_dev)
                spatial_inds_batch[...] = spatial_inds[:, None, r1:r2]
                temporal_feat_batch[...] = temporal_features[:, t1:t2, None]
                mat = self.matrix_access(spatial_inds_batch, temporal_feat_batch)
                output[:, r1:r2] += temporal_input_flt[:, t1:t2] @ mat.conj()
                
        return output.view(N, *self.im_size)

    def forward_matrix_prod(self,
                            spatial_input: torch.Tensor) -> torch.Tensor:
        """
        Given spatial input vector input(r) returns
        
        out(t) = int_r W(r, p(t)) * input(r) dr
        (same as 'A * x')
        
        Parameters:
        -----------
        spatial_input: torch.Tensor
            the spatial input vector with shape (N, *im_size)
        
        Returns:
        --------
        torch.Tensor
            the temporal output vector with shape (N, *trj_size)
        """
        temporal_features = self.temporal_features((slice(None),))
        return self._continuous_forward_matrix_prod(spatial_input, temporal_features)

    def adjoint_matrix_prod(self,
                            temporal_input: torch.Tensor) -> torch.Tensor:
        """
        Given temporal input vector input(t) returns
        out(r) = int_t conj(W(r, p(t))) * input(t) dt
        (same as 'A^H * x')
        
        Parameters:
        -----------
        temporal_input: torch.Tensor
            the temporal input vector with shape (N, *trj_size)
        
        Returns:
        --------
        torch.Tensor
            the spatial output vector with shape (N, *im_size)
        """
        temporal_features = self.temporal_features((slice(None),))
        return self._continuous_adjoint_matrix_prod(temporal_input, temporal_features)

    def gram_spatial_prod(self,
                          spatial_input: torch.Tensor) -> torch.Tensor:
        """
        Given spatial input vector input(r) returns

        out = adjoint(forward(input))
        out(r) = int_t conj(W(r, p(t))) * (int_r0 W(r0, p(t)) * input(r0) dr0) dt
        (same as 'A^H * A * x')

        Parameters:
        -----------
        spatial_input: torch.Tensor
            the spatial input vector with shape (N, *im_size)
        
        Returns:
        --------
        torch.Tensor
            the spatial output vector with shape (N, *im_size)
        """
        return self.adjoint_matrix_prod(self.forward_matrix_prod(spatial_input))
    
    def gram_temporal_prod(self,
                           temporal_input: torch.Tensor) -> torch.Tensor:
        """
        Given temporal input vector input(t) returns
        
        out = forward(adjoint(input))
        out(t) = int_r W(r, p(t)) * (int_t0 conj(W(r, p(t0))) * input(t0) dt0) dr
        (same as 'A * A^H * x')

        Parameters:
        -----------
        temporal_input: torch.Tensor
            the temporal input vector with shape (N, *trj_size)
        
        Returns:
        --------
        torch.Tensor
            the temporal output vector with shape (N, *trj_size)
        """
        return self.forward_matrix_prod(self.adjoint_matrix_prod(temporal_input))
    
    def test_matrix_prods(self,
                          spatial_batch_size: Optional[int] = 2**12,
                          temporal_batch_size: Optional[int] = 2**12):
        """
        Tests the forward, adjoint, forward-adjoint, and adjoint-forward matrix products.

        Parameters:
        -----------
        spatial_batch_size: int
            the batch size for spatial indices
        temporal_batch_size: int
            the batch size for temporal indices

        Prints:
        -------
        the relative error of the matrix products.
        """

        # Consts
        im_size = self.im_size
        trj_size = self.trj_size
        d = len(im_size)
        k = len(trj_size)

        # Gen Inputs
        inp_spatial = torch.randn(self.im_size, device=self.torch_dev, dtype=complex_dtype)
        inp_spatial_flt = inp_spatial.flatten()
        inp_temporal = torch.randn(self.trj_size, device=self.torch_dev, dtype=complex_dtype)
        inp_temporal_flt = inp_temporal.flatten()

        # Indices for matrix access
        temporal_inds = gen_grd(self.trj_size, self.trj_size) + torch.tensor(self.trj_size) // 2
        temporal_inds = temporal_inds.reshape((-1, k)).T.type(torch.int)
        temporal_features = self.temporal_features(temporal_inds)
        f = temporal_features.shape[-1]
        spatial_inds = gen_grd(self.im_size, self.im_size) + torch.tensor(self.im_size) // 2
        spatial_inds = spatial_inds.reshape((-1, len(self.im_size))).T.type(torch.int)

        # Matrix products
        Ax = self.forward_matrix_prod(inp_spatial)
        AHy = self.adjoint_matrix_prod(inp_temporal)
        AHAx = self.forward_adjoint_matrix_prod(inp_spatial)
        AAHy = self.adjoint_forward_matrix_prod(inp_temporal)

        # Ground truth matrix products
        Ax_gt = torch.zeros_like(inp_temporal_flt)
        AHy_gt = torch.zeros_like(inp_spatial_flt)
        for r1, r2 in batch_iterator(inp_spatial_flt.shape[0], spatial_batch_size):
            for t1, t2 in batch_iterator(inp_temporal_flt.shape[0], temporal_batch_size):
                spatial_inds_batch = torch.zeros((d, t2-t1, r2-r1), dtype=torch.int, device=self.torch_dev)
                temporal_feat_batch = torch.zeros((t2-t1, r2-r1, f), dtype=torch.int, device=self.torch_dev)
                spatial_inds_batch[...] = spatial_inds[:, None, r1:r2]
                temporal_feat_batch[...] = temporal_features[t1:t2, None, :]
                mat = self.matrix_access(spatial_inds_batch, temporal_feat_batch)
                Ax_gt[t1:t2] += einsum(mat, inp_spatial_flt[r1:r2], 'T R, R -> T') 
                AHy_gt[r1:r2] += einsum(mat.conj(), inp_temporal_flt[t1:t2], 'T R, T -> R')
        AHAx_gt = torch.zeros_like(inp_spatial_flt)
        AAHy_gt = torch.zeros_like(inp_temporal_flt)
        for r1, r2 in batch_iterator(inp_spatial_flt.shape[0], spatial_batch_size):
            for t1, t2 in batch_iterator(inp_temporal_flt.shape[0], temporal_batch_size):
                spatial_inds_batch = torch.zeros((d, t2-t1, r2-r1), dtype=torch.int, device=self.torch_dev)
                temporal_feat_batch = torch.zeros((t2-t1, r2-r1, f), dtype=torch.int, device=self.torch_dev)
                spatial_inds_batch[...] = spatial_inds[:, None, r1:r2]
                temporal_feat_batch[...] = temporal_features[t1:t2, None, :]
                mat = self.matrix_access(spatial_inds_batch, temporal_feat_batch)
                AAHy_gt[t1:t2] += einsum(mat, AHy_gt[r1:r2], 'T R, R -> T') 
                AHAx_gt[r1:r2] += einsum(mat.conj(), Ax_gt[t1:t2], 'T R, T -> R')

        # Compute errors
        Ax_err = torch.norm(Ax.flatten() - Ax_gt) / torch.norm(Ax_gt)
        AHy_err = torch.norm(AHy.flatten() - AHy_gt) / torch.norm(AHy_gt)
        AHAx_err = torch.norm(AHAx.flatten() - AHAx_gt) / torch.norm(AHAx_gt)
        AAHy_err = torch.norm(AAHy.flatten() - AAHy_gt) / torch.norm(AAHy_gt)
        print(f'Ax Error = {Ax_err.item()}')
        print(f'AHy Error = {AHy_err.item()}')
        print(f'AHAx Error = {AHAx_err.item()}')
        print(f'AAHy Error = {AAHy_err.item()}')


# TODO FIXME 
class B0(spatio_temporal):
    """
    This spatial-temporal imperfection has the form

    W(r, p(t)) = exp(-j 2pi B0(r) * t)
    p(t) = t
    """

    def __init__(self, 
                 im_size: tuple, 
                 trj_size: tuple,
                 readout_dim: int,
                 b0_map: torch.Tensor):
        """
        Parameters:
        -----------
        b0 : torch.Tensor
            the B0 field map in CYCLES/SAMPLE with shape (*im_size)
        readout_dim : int
            specifies which dimension of trj_size is the readout dimension
        """
        trj_size = [1 if i != readout_dim else trj_size[i] for i in range(len(trj_size))]
        super(B0, self).__init__(im_size, trj_size, device=b0_map.device)

        # Store the B0 map and readout dimension
        self.b0_map = b0_map
        self.readout_dim = readout_dim
        
        # Create samples in the readout dimension
        num_readout = trj_size[readout_dim]
        self.samps = torch.arange(num_readout, device=b0_map.device)

        # Create a NUFFT for later use -- b0 matrix prod is fourier in time!
        assert num_readout % 2 == 0, 'Even readout dim only'
        self.nufft = sigpy_nufft(im_size=(num_readout,))
        self.num_readout = num_readout
        self.num_vox = torch.prod(torch.tensor(im_size)).item()

        # Compute phase_0 to make sure that no phase is applied at first time point
        self.phase_0 = torch.exp(2j * torch.pi * b0_map * num_readout/2) * (num_readout ** 0.5)
    
        # Toeplitz kernels for temporal gram operator
        weights = self.phase_0 * self.phase_0.conj()
        trj = (-b0_map * num_readout)[..., None]
        self.toeplitz_kerns = torch.zeros((num_readout * 2), device=b0_map.device, dtype=complex_dtype)
        self.toeplitz_kern = self.nufft.calc_teoplitz_kernels(trj[None,], weights)
                    
    def temporal_features(self,
                          temporal_inds: torch.Tensor) -> torch.Tensor:
            return self.samps[temporal_inds[self.readout_dim]][None,] # single feature

    def get_temporal_clusters(self,
                              num_clusters: int) -> torch.Tensor:
        # Uniform clustering
        clusters, inds_ro = quantize_data(self.samps[:, None], num_clusters, method='uniform')

        # Reshape inds to size of trajectory 
        inds = torch.zeros(self.trj_size, device=self.b0_map.device, dtype=torch.int)
        tup = (None,) * self.readout_dim + (slice(None),) + (None,) * (len(self.trj_size) - self.readout_dim - 1)
        inds[...] = inds_ro[tup]

        return clusters.T, inds

    def matrix_access(self,
                      spatial_inds: torch.Tensor,
                      temporal_features: torch.Tensor) -> torch.Tensor:
        b0 = self.b0_map[tuple(spatial_inds)]
        return torch.exp(-2j * torch.pi * b0 * temporal_features[0])
    
    def forward_matrix_prod(self,
                            spatial_input: torch.Tensor) -> torch.Tensor:
        
        # Adjoint has positive exponent so multiply b0 by -1
        freqs = (-self.b0_map * self.num_readout)[..., None]
        out_readout = self.nufft.adjoint((spatial_input)[None,] * self.phase_0.conj(), freqs[None,])[0]

        # Fill in other dimensions
        temporal_output = torch.zeros(self.trj_size, device=self.b0_map.device, dtype=complex_dtype)
        tup = (slice(None),) + (None,) * self.readout_dim + (slice(None),) + (None,) * (len(self.trj_size) - self.readout_dim - 1)
        temporal_output[...] = out_readout[tup]

        return temporal_output
    
    def adjoint_matrix_prod(self,
                            temporal_input: torch.Tensor) -> torch.Tensor:
        # Forward nufft has negative exponent so multiple b0 by -1
        freqs = (-self.b0_map * self.num_readout)[..., None]
        
        # Select only readout dim
        tup = (slice(None),) + (0,) * self.readout_dim + (slice(None),) + (0,) * (len(self.trj_size) - self.readout_dim - 1)
        temporal_readout = temporal_input[tup]

        imgs = self.nufft(temporal_readout[None,], freqs[None,])[0] * self.phase_0
        return (self.mps.conj() * imgs).sum(dim=0)
    
    def adjoint_forward_matrix_prod(self,
                                    temporal_input: torch.Tensor) -> torch.Tensor:
        # Select readout dim only
        tup = (slice(None),) + (0,) * self.readout_dim + (slice(None),) + (0,) * (len(self.trj_size) - self.readout_dim - 1)
        out_readout = temporal_input[tup]
        
        # Zero padded FFTs
        padder = PadLast((self.toeplitz_kerns.shape[-1],), (self.num_readout,))
        out_readout = padder.forward(out_readout)

        # FFT, matmul with toeplitz kernels, iFFT
        out_readout = fft(out_readout, dim=-1)
        out_readout = einsum(out_readout, self.toeplitz_kerns, 'Ci T, Ci Co T -> Co T')
        out_readout = ifft(out_readout, dim=-1)

        # Crop
        out_readout = padder.adjoint(out_readout)
        
        # Fill in other dimensions
        temporal_output = torch.zeros((self.mps.shape[0], *self.trj_size), device=self.b0_map.device, dtype=complex_dtype)
        tup = (slice(None),) + (None,) * self.readout_dim + (slice(None),) + (None,) * (len(self.trj_size) - self.readout_dim - 1)
        temporal_output[...] = out_readout[tup]
        return temporal_output

class high_order_phase(spatio_temporal):
    """
    This spatial-temporal imperfection has the form

    W(r, p(t)) = exp(-j 2pi * sum_b phi_b(r) * alpha_b(t))
    p(t) = [alpha_1(t), alpha_2(t), ..., alpha_B(t)]

    phi_b(r) are usually polynomials/spherical harmonics in r.
    """

    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 spatial_batch_size: Optional[int] = None,
                 temporal_batch_size: Optional[int] = None):
        """
        Parameters:
        -----------
        phis : torch.Tensor
            the spatial phase maps with shape (B, *im_size)
        alphas : torch.Tensor
            the temporal phase coefficents with shape (B, *trj_size)
        """
        # Store
        self.phis = phis
        self.alphas = alphas
        self.B, *trj_size = alphas.shape
        B, *im_size = phis.shape
        assert alphas.device == phis.device, 'phis and alphas must be on same device'
        assert B == self.B, 'Number of spatial and temporal features must match'
        
        # ------------- High Order Phase Via NUFFT -------------
        # Params
        W = 4
        os = 1.25
        
        # Scale alphas so that phis are between [-1/2, 1/2]
        self.scales = phis.abs().view(self.B, -1).max(dim=-1).values * 2 # B 
        self.alphas_rescaled = self.alphas.moveaxis(0, -1) * self.scales
        
        # Determine matrix size from extent of scaled alphas
        alphas_rescaled_max = self.alphas_rescaled.abs().view((-1, self.B)).max(dim=0).values
        matrix_size = ((alphas_rescaled_max + W/os/2) * 2).ceil().int()
        # TODO FIXME. Currently uses isotropic matrix size, would be nice if we can fix that.
        # This is because we need that each matrix dimension after multiplying by os is an integer.
        mx = matrix_size.max().item()
        matrix_size = (matrix_size.max() * (matrix_size * 0 + 1)).int().tolist()
        os = round(mx * os) / mx
        matrix_size_os = [ceil(matrix_size[i] * os) for i in range(len(matrix_size))]
        print(f'Required Grid Size = {matrix_size_os}')
        
        # Create NUFFT module and define desired alphas grid points + apodization
        self.nufft = sigpy_nufft(matrix_size, oversamp=os, width=W)
        self.alphas_grid = ((gen_grd(matrix_size_os, matrix_size_os).to(phis.device) / os) / self.scales).moveaxis(-1, 0)
        self.apod = torch.prod(torch.stack([self.nufft._apodize_1d(phis[i] / self.scales[i]) for i in range(self.B)], dim=0), dim=0)
        
        
        # Find only required alphas on the grid and correspinding indices
        kern = gen_grd([W]*self.B, [W]*self.B).view(-1, self.B).to(phis.device) / os
        alphas_required = (self.alphas_rescaled * os).round().view(-1,self.B) / os
        alphas_required = alphas_required.unique(dim=0)[:, None, :] + kern
        alphas_required = alphas_required.reshape(-1, self.B).unique(dim=0)
        inds_required = alphas_required * os
        breakpoint()
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        alphas = (self.alphas.moveaxis(0, -1) * self.scales).moveaxis(-1, 0).cpu()
        alphas_grid = (self.alphas_grid.moveaxis(0, -1) * self.scales).moveaxis(-1, 0).cpu()
        
        ax.scatter(*alphas.view(3,-1), marker='.', alpha=0.3)
        # ax.scatter(*alphas_grid.view(3,-1), marker='.', alpha=0.05)
        plt.show()
        quit()

        super(high_order_phase, self).__init__(im_size, trj_size, 
                                               spatial_batch_size=spatial_batch_size, temporal_batch_size=temporal_batch_size, 
                                               device=alphas.device)
    
    def get_temporal_clusters(self,
                              num_clusters: int) -> Union[torch.Tensor, torch.Tensor]:
            """
            Returns clusters and indices of temporal features.
    
            Parameters:
            -----------
            num_clusters: int
                the number of clusters
            
            Returns:
            --------
            clusters: torch.Tensor
                the temporal clusters with shape (f, num_clusters)
            inds: torch.Tensor <int>
                the indices of the temporal clusters with shape (*trj_size), values in [0, num_clusters-1]
            """
            clusters, inds = quantize_data(self.alphas_rescaled, num_clusters, method='cluster')
            clusters = clusters / self.scales
            return clusters.T, inds
    
    def matrix_access(self,
                      spatial_inds: torch.Tensor,
                      temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Returns the value of W(r, p(t)) at the given indices.

        Parameters:
        -----------
        spatial_inds: torch.Tensor <int>
            the spatial indices with shape (d, ...)
        temporal_features: torch.Tensor <int>
            the phase coefficients with shape (B, ...)
            same as p(t)
        
        Returns:
        --------
        torch.Tensor
            the value of e^{-j 2pi phi(r[spatial_inds]) * temporal_features} shape (...)
        """
        phis = self.phis[(slice(None),) + tuple(spatial_inds)] # B ...
        return torch.exp(-2j * torch.pi * (phis * temporal_features).sum(dim=0))
    
    def temporal_features(self, 
                          temporal_inds: torch.Tensor) -> torch.Tensor:
        """
        Returns the temporal features p(t) at the given indices.

        Parameters:
        -----------
        temporal_inds: torch.Tensor <int>
            the temporal index with shape (k, ...)
        
        Returns:
        --------
        torch.Tensor
            the temporal features p(t) at the given indices with shape (f, ...)
            f is the number of temporal features
            same as p(t)
        """
        return self.alphas[(slice(None),) + tuple(temporal_inds)]

    def forward_matrix_prod(self, spatial_input):
        # return super().forward_matrix_prod(spatial_input)
        matmul = self._continuous_forward_matrix_prod(spatial_input * self.apod, self.alphas_grid) # N ...
        out = self.nufft.forward_interp_only(matmul, self.alphas_rescaled[None,])

        return out
    
    def adjoint_matrix_prod(self, temporal_input):
        # return super().adjoint_matrix_prod(temporal_input)
        kgrid = self.nufft.adjoint_grid_only(temporal_input, self.alphas_rescaled[None,])
        out = self._continuous_adjoint_matrix_prod(kgrid, self.alphas_grid) * self.apod
        
        return out