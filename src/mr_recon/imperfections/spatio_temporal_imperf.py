import torch

from typing import Optional, Union
from einops import rearrange, einsum
from mr_recon.dtypes import complex_dtype
from mr_recon.pad import PadLast
from mr_recon.fourier import sigpy_nufft, fft, ifft
from mr_recon.utils import gen_grd, quantize_data, batch_iterator
from mr_recon.spatial import spatial_resize

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
            the temporal features p(t) at the given indices with shape (..., f)
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
            the temporal clusters with shape (num_clusters, f)
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
            the temporal features with shape (..., f)
            same as p(t)
        
        Returns:
        --------
        torch.Tensor
            the value of W(r, p(t)) at the given indices with shape (...)
        """
        raise NotImplementedError

    def forward_matrix_prod(self,
                            spatial_input: torch.Tensor) -> torch.Tensor:
        """
        Applies W(r, p(t)) to spatial input vector:

        out(t) = int_r W(r, p(t)) * input(r) dr

        Parameters:
        -----------
        spatial_input: torch.Tensor
            the spatial input vector with shape (*im_size)
        
        Returns:
        --------
        torch.Tensor
            the output temporal vector with shape (*trj_size)
        """
        raise NotImplementedError

    def adjoint_matrix_prod(self,
                            temporal_input: torch.Tensor) -> torch.Tensor:
        """
        Applies W(r, p(t)) to temporal input vector:

        out(r) = int_t conj(W(r, p(t))) * input(t) dt

        Parameters:
        -----------
        temporal_input
            the input temporal vector with shape (N, *trj_size)
        
        Returns:
        --------
        torch.Tensor
            the spatial output vector with shape (N, *im_size)
        """
        raise NotImplementedError

    def forward_adjoint_matrix_prod(self,
                                    spatial_input: torch.Tensor) -> torch.Tensor:
        """
        Given spatial input vector input(r) returns

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
    
    def adjoint_forward_matrix_prod(self,
                                    temporal_input: torch.Tensor) -> torch.Tensor:
        """
        Given temporal input vector input(t) returns

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
        quit()

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
                 b0_map: torch.Tensor,
                 coil_maps: Optional[torch.Tensor] = None):
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

        if coil_maps is None:
            coil_maps = torch.ones((1, *im_size), device=b0_map.device, dtype=complex_dtype)
        assert coil_maps.shape[1:] == im_size, 'Coil maps must have same size as image'

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
        # self.toeplitz_kerns = self.nufft.calc_teoplitz_kernels((-b0_map * num_readout)[None, ..., None], weights[None,])
        C = coil_maps.shape[0]
        trj = (-b0_map * num_readout)[..., None]
        self.toeplitz_kerns = torch.zeros((C, C, num_readout * 2), device=b0_map.device, dtype=complex_dtype)
        for c1 in range(C):
            weights_c1c2 = weights * coil_maps * coil_maps[c1].conj()
            kern = self.nufft.calc_teoplitz_kernels(trj[None,], weights_c1c2)
            self.toeplitz_kerns[c1, :, ...] = kern
        self.mps = coil_maps
                    
    def temporal_features(self,
                          temporal_inds: torch.Tensor) -> torch.Tensor:
            return self.samps[temporal_inds[self.readout_dim]][..., None] # single feature

    def get_temporal_clusters(self,
                              num_clusters: int) -> torch.Tensor:
        # Uniform clustering
        clusters, inds_ro = quantize_data(self.samps[:, None], num_clusters, method='uniform')

        # Reshape inds to size of trajectory 
        inds = torch.zeros(self.trj_size, device=self.b0_map.device, dtype=torch.int)
        tup = (None,) * self.readout_dim + (slice(None),) + (None,) * (len(self.trj_size) - self.readout_dim - 1)
        inds[...] = inds_ro[tup]

        return clusters, inds

    def matrix_access(self,
                      spatial_inds: torch.Tensor,
                      temporal_features: torch.Tensor) -> torch.Tensor:
        b0 = self.b0_map[tuple(spatial_inds)]
        mps = self.mps[(slice(None),) + tuple(spatial_inds)]
        return torch.exp(-2j * torch.pi * b0 * temporal_features[..., 0]) * mps
    
    def forward_matrix_prod(self,
                            spatial_input: torch.Tensor) -> torch.Tensor:
        
        # Adjoint has positive exponent so multiply b0 by -1
        freqs = (-self.b0_map * self.num_readout)[..., None]
        out_readout = self.nufft.adjoint((spatial_input * self.mps)[None,] * self.phase_0.conj(), freqs[None,])[0]

        # Fill in other dimensions
        temporal_output = torch.zeros((self.mps.shape[0], *self.trj_size), device=self.b0_map.device, dtype=complex_dtype)
        tup = (slice(None),) + (None,) * self.readout_dim + (slice(None),) + (None,) * (len(self.trj_size) - self.readout_dim - 1)
        # temporal_output = torch.zeros(self.trj_size, device=self.b0_map.device, dtype=complex_dtype)
        # tup = (None,) * self.readout_dim + (slice(None),) + (None,) * (len(self.trj_size) - self.readout_dim - 1)
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
                 im_size: tuple,
                 trj_size: tuple,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 num_alpha_clusters: Optional[int] = None,
                 img_downsample_factor: Optional[int] = 1,):
        """
        Parameters:
        -----------
        phis : torch.Tensor
            the spatial phase maps with shape (B, *im_size)
        alphas : torch.Tensor
            the temporal phase maps with shape (B, *trj_size)
        num_alpha_clusters : int
            the number of temporal clusters to use 
        img_downsample_factor : int
            the factor to fourier downsample the spatial phase maps

        """
        B = alphas.shape[0]
        assert phis.shape[0] == B, 'Number of spatial and temporal basis functions must match'
        assert phis.device == alphas.device, 'Spatial and temporal basis functions must be on the same device'

        # Store temporal clusters for faster operator application
        if num_alpha_clusters is not None:
            self.betas, idxs = quantize_data(alphas.reshape((B, -1)).T, num_alpha_clusters, method='cluster')
            self.betas = self.betas.T # (B, num_alpha_clusters)
            self.idxs = idxs.reshape(trj_size)
        else:
            self.betas = alphas.reshape((B, -1))
            self.idxs = torch.arange(self.betas.shape[-1], device=alphas.device).reshape(trj_size)

        # Spatial downsampling
        downsampled_shape = tuple([s // img_downsample_factor for s in im_size])
        method = 'bilinear'
        self.downsample = lambda x : spatial_resize(x, downsampled_shape, method).type(x.dtype)
        self.upsample = lambda x : spatial_resize(x, im_size, method).type(x.dtype)
        self.phis = self.downsample(phis)
        self.scale = img_downsample_factor ** len(im_size)

        # Use downsampled image and number of clusters for dimensions
        super(high_order_phase, self).__init__(downsampled_shape, (self.betas.shape[-1],), device=phis.device)
        
    def temporal_features(self,
                          temporal_inds: torch.Tensor) -> torch.Tensor:
        features = self.betas[(slice(None),) + tuple(temporal_inds)].moveaxis(0, -1)
        return features

    def get_temporal_clusters(self, 
                              num_clusters: int) -> torch.Tensor:
        # First grab all temporal features
        trj_size = self.trj_size
        temporal_inds = (slice(None),) * len(trj_size)
        temporal_features = self.temporal_features(temporal_inds)
        clusters, inds = quantize_data(temporal_features, num_clusters, method='cluster')
        return clusters, inds

    def matrix_access(self,
                      spatial_inds: torch.Tensor,
                      temporal_features: torch.Tensor) -> torch.Tensor:
        phis = self.phis[(slice(None),) + tuple(spatial_inds)]
        return torch.exp(-2j * torch.pi * einsum(phis, temporal_features, 'B ..., ... B -> ...'))
    
    def forward_matrix_prod(self,
                            spatial_input: torch.Tensor) -> torch.Tensor:
        phis = self.phis
        matrix = torch.exp(-2j * torch.pi * einsum(phis, self.betas, 'B ..., B K -> ... K'))
        return (matrix * spatial_input[..., None]).sum(dim=tuple(range(-len(self.im_size)-1, -1)))
        # return einsum(matrix, spatial_input, '... K, ... -> K')
    
    def adjoint_matrix_prod(self,
                            temporal_input: torch.Tensor) -> torch.Tensor:
        phis = self.phis
        matrix = torch.exp(2j * torch.pi * einsum(phis, self.betas, 'B ..., B K -> K ...'))
        tup = (slice(None),) * temporal_input.ndim + (None,) * len(self.im_size)
        return (matrix * temporal_input[tup]).sum(dim=-len(self.im_size)-1)
        # return einsum(matrix, temporal_input, 'K ..., K -> ...')