import torch

from mr_recon.fourier import NUFFT
from mr_recon.fourier.common import calc_toep_kernel_helper
from mr_recon.utils import np_to_torch
from typing import Optional
from torchkbnufft import KbNufft, KbNufftAdjoint

class torchkb_nufft(NUFFT):

    def __init__(self,
                 im_size: tuple,
                 torch_dev: Optional[torch.device] = torch.device('cpu'),
                 oversamp: Optional[float] = 2.0,
                 numpoints: Optional[int] = 6):
        super().__init__(im_size)
        
        im_size_os = tuple([round(i * oversamp) for i in im_size])
        self.kb_ob = KbNufft(im_size, device=torch_dev, grid_size=im_size_os, numpoints=numpoints).to(torch_dev)
        self.kb_adj_ob = KbNufftAdjoint(im_size, device=torch_dev, grid_size=im_size_os, numpoints=numpoints).to(torch_dev)
        self.oversamp = oversamp
        self.numpoints = numpoints

    def rescale_trajectory(self,
                           trj: torch.Tensor) -> torch.Tensor:
        
        # Rescale to -pi, pi
        im_size_arr = torch.tensor(self.im_size).to(trj.device)
        tup = (None,) * (trj.ndim - 1) + (slice(None),)
        trj_rs = torch.pi * trj / (im_size_arr[tup] / 2)

        return trj_rs.type(torch.float32)

    def forward(self, 
                img: torch.Tensor, 
                trj: torch.Tensor) -> torch.Tensor:
        
        # To torch
        img_torch, trj_torch = np_to_torch(img, trj)
        im_size = self.im_size
        N = trj.shape[0]
        d = trj.shape[-1]

        # Reshape - NUFFT - Reshape
        img_torchkb = img_torch.reshape((N, -1, *im_size))
        omega = trj_torch.reshape((N, -1, d)).swapaxes(-2, -1)
        ksp = self.kb_ob(image=img_torchkb, omega=omega, norm='ortho')
        ksp = ksp.reshape((N, *img.shape[1:-d], *trj.shape[1:-1]))
        return ksp * self.oversamp

    def adjoint(self,
                ksp: torch.Tensor,
                trj: torch.Tensor) -> torch.Tensor:

        # To torch
        ksp_torch, trj_torch = np_to_torch(ksp, trj)
        im_size = self.im_size
        N = trj.shape[0]
        d = trj.shape[-1]

        # Reshape - NUFFT - Reshape
        ksp_torch_kb = ksp_torch.reshape((N, -1, *trj.shape[1:-1]))
        ksp_torch_kb = ksp_torch_kb.reshape((N, ksp_torch_kb.shape[1], -1))
        omega = trj_torch.reshape((N, -1, d)).swapaxes(-2, -1)
        img = self.kb_adj_ob(data=ksp_torch_kb, omega=omega, norm='ortho')
        img = img.reshape((N, *ksp.shape[1:-(trj.ndim - 2)], *im_size))
        return img * self.oversamp
    
    def calc_teoplitz_kernels(self,
                              trj: torch.Tensor,
                              weights: Optional[torch.Tensor] = None,
                              os_factor: Optional[float] = 2.0,):
        """
        Calculate the Toeplitz kernels for the NUFFT

        Parameters:
        -----------
        trj : torch.Tensor <float>
            input trajectory with shape (N, *trj_batch, len(im_size))
        weights : torch.Tensor <float>
            weighting function with shape (N, *trj_batch)
        os_factor : float
            oversampling factor for toeplitz

        Returns:
        --------
        toeplitz_kernels : torch.Tensor <complex>
            the toeplitz kernels with shape (N, *im_size_os)
            where im_size_os is the oversampled image size
        """

        # Consts
        im_size_os = tuple([round(i * os_factor) for i in self.im_size])

        # Make new instance of NUFFT with oversampled image size
        nufft_os = torchkb_nufft(im_size_os, torch_dev=trj.device, oversamp=self.oversamp, numpoints=self.numpoints)

        return calc_toep_kernel_helper(nufft_os.adjoint, trj, weights) * (os_factor ** len(self.im_size))
  