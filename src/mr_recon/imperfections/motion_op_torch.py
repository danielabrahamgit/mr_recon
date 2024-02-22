# +
import sigpy as sp
from sigpy.linop import Linop 
import copy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import copy

def generate_grids(ishape):
    """
    Generate spatio, temporal and spatio-temporal grids for a cubic image of shape Nkm.
    
    ishape : tuple
        input array shape, make sure dimension sizes are even
    
    """
    rGrids = []
    kGrids = []
    rkGrids = []
    for ax_ in range(len(ishape)):
        min_k = -np.floor(ishape[ax_] / 2) 
        max_k = np.ceil(ishape[ax_] / 2) 
        Nk = max_k - min_k
        
        r_m = torch.arange(min_k, max_k, 1)
        if len(r_m)== 0:
            r_m = torch.Tensor([0.])
        kGrids.append(2 / Nk * np.pi * r_m)
        rGrids.append(r_m)

    per =  [ [0, 2, 1], [1, 0, 2] ]
    for ii in range(2):
        rk = []
        for jj in range(3):
            rk.append( torch.outer(kGrids[per[ii][jj]], rGrids[per[1-ii][jj]]) )
        rkGrids.append(rk)
        
    return rGrids, kGrids, rkGrids

class rigid_motion(nn.Module):

    def __init__(self, 
                 ishape: tuple,
                 pad: Optional[bool] = False):
        """
        Sinc-interpolated 2D or 3D rotation operating in k-space.

        ishape : tuple dimensions [batch_size, x, y(, z)]
            input images are expected to be 2d or 3d tensors
        pad : bool, optional 
            whether to pad the input image to get a cubic tensor
        """
        super(rigid_motion, self).__init__()
    
        self.pad = False
        if (len(ishape) == 2):
            # add mock batch dimension
            ishape = (1, ishape[0], ishape[1])
        if (len(ishape) == 3):
            if (ishape[-1] != ishape[-2]):
                print('WARNING: input array will be padded to get a square array')
                self.pad = True
            self.padshape = (max(ishape), max(ishape), 1)
        elif (len(ishape) == 4):
            if (ishape[-1] != ishape[-2] or ishape[-2] != ishape[-3]):
                print('WARNING: input array will be padded to get a cubic array')
                self.pad = True
            self.padshape = (max(ishape), max(ishape), max(ishape))
        else:
            raise Exception('Can transform only 2d or 3d images, sorry')

        self.oshape = ishape[1:] # xy(z)
        self.Nk = max(self.oshape)
        
        # save the grids as they'll be used for all motion params
        self.rGrids, self.kGrids, self.rkGrids = generate_grids(self.padshape)
    
    def _reset_motion_params(self, 
                             motion_params: torch.Tensor):
        """
        motion_params - Tensor of shape (N_shots, 6) - translations and rotations
            translations are in [pixels]
            rotation angles are in [rads]
        """
        self.do_rot = False
        self.do_transl = False
    
        if motion_params.ndim == 1 and motion_params.shape[0] == 6:
            self.mot_traj = motion_params[None, :]
        elif motion_params.ndim == 2:
            self.mot_traj = motion_params
        else:
            raise Exception('Motion parameters tensor should be of size (N_shots, N_motion_params)')
    
        self.device = self.mot_traj.device
        thetas = self.mot_traj[:,-3:] * torch.Tensor([-1, 1, -1])[None, :].to(self.device) # sign indicates the rotation orientation
        if (thetas.any()):
            self.do_rot = True
            # precompute the shear matrices    
            tantheta2 = 1j * torch.tan(thetas/2)
            sintheta = -1j * torch.sin(thetas)
            
            self.V_mat_tan_xyzr = [] # of size 3 - for each rotation axis, each element has size (rkGrid_size)
            self.V_mat_sin_xyzr = []
            for ax_ in range(3):
                exp_ = torch.exp(tantheta2[None, None, :, 2-ax_] * self.rkGrids[0][ax_][:,:, None].to(self.device)) 
                self.V_mat_tan_xyzr.append(exp_)

                exp_ = torch.exp(sintheta[None, None, :, 2-ax_] * self.rkGrids[1][ax_][:,:, None].to(self.device))
                self.V_mat_sin_xyzr.append(exp_)  
                
        shifts = self.mot_traj[:,:3] 
        if (shifts.any()):
            self.do_transl = True

        if self.do_transl:
            kxx, kyy, kzz = [tensor.to(self.device) for tensor in \
                             torch.meshgrid(self.kGrids[0], self.kGrids[1], self.kGrids[2], indexing='ij')]
            dim_k = self.kGrids[0].shape[0]
            # dx is along y in Python, dy is along x and with the opposite sign
            self.U_mat_xyzr = torch.exp(-1j * ( -kxx[...,None] * shifts[None, None, None,:,0] + \
                                                 kyy[...,None] * shifts[None, None, None,:,1] + \
                                                 kzz[...,None] * shifts[None, None, None,:,2])) 
            
    def _apply(self, 
               input: torch.Tensor):
        """
        img : 2d-array and 3d-tensors
            dimensions xy or xyz - coil and spatial

        Return 
            image_tr : ishape tensor
            Warped image        
        """
        assert torch.is_complex(input)
        assert input.device == self.device

        image_in = copy.deepcopy(input) # bxy(z)
        if image_in.ndim == 2:
            image_in = image_in[None, ...]
        if image_in.ndim == 4:
            self.rot_axes = [0, 1, 2]
            image_xyzr = image_in[..., None] # r - repetition or shot dimension
        elif image_in.ndim == 3:
            self.rot_axes = [0]
            image_xyzr = image_in[..., None, None]
        
        if self.pad:
            pad = [0, 0]
            for ax_ in range(len(self.oshape)-1, -1, -1):
                diff = self.padshape[ax_] - self.oshape[ax_]
                pad.append(int(np.floor(diff/2)))
                pad.append(int(np.ceil(diff/2)))
            image_xyzr = F.pad(input=image_xyzr, pad=tuple(pad), mode='constant', value=0)
        
        if self.do_rot:
            per =  [ [1, 3, 2], [2, 1, 3] ] # one higher than corresponding to yzx
            for axis in self.rot_axes:
                newaxis = 2 - axis + 1
                V_tan_exp_xyzr = torch.unsqueeze(torch.unsqueeze(self.V_mat_tan_xyzr[axis], dim=0), dim=newaxis)
                V_sin_exp_xyzr = torch.unsqueeze(torch.unsqueeze(self.V_mat_sin_xyzr[axis], dim=0), dim=newaxis)
                image_xyzr = torch.fft.fftshift(torch.fft.fft(image_xyzr, dim=per[0][axis]), dim=per[0][axis])
                image_xyzr =  V_tan_exp_xyzr * image_xyzr 
                image_xyzr = torch.fft.ifft(torch.fft.ifftshift(image_xyzr, dim=per[0][axis]), dim=per[0][axis])
                
                image_xyzr = torch.fft.fftshift(torch.fft.fft(image_xyzr, dim=per[1][axis]), dim=per[1][axis])
                image_xyzr = (V_sin_exp_xyzr * image_xyzr)
                image_xyzr = torch.fft.ifft(torch.fft.ifftshift(image_xyzr, dim=per[1][axis]), dim=per[1][axis])

                image_xyzr = torch.fft.fftshift(torch.fft.fft(image_xyzr, dim=per[0][axis]), dim=per[0][axis])
                image_xyzr = (V_tan_exp_xyzr * image_xyzr)
                image_xyzr = torch.fft.ifft(torch.fft.ifftshift(image_xyzr, dim=per[0][axis]), dim=per[0][axis])        
        
        if self.do_transl:
            axes_ = (1, 2, 3) # hard code
            image_xyzr = torch.fft.ifftn( torch.fft.ifftshift( self.U_mat_xyzr * torch.fft.fftshift(torch.fft.fftn\
                                    ( image_xyzr, dim=axes_), \
                                    dim=axes_ ), dim=axes_ ), dim=axes_ )   
             
        if self.pad:
            if len(self.oshape) == 3:
                image_xyzr = image_xyzr[:,pad[-2]:self.Nk-pad[-1], pad[0]:self.Nk-pad[1], 0, :]
            elif len(self.oshape) == 4:
                image_xyzr = image_xyzr[:,pad[-2]:self.Nk-pad[-1], pad[-4]:self.Nk-pad[-3], pad[-6]:self.Nk-pad[-5], :]
                
        return image_xyzr


