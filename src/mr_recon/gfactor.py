import torch

from tqdm import tqdm
from typing import Optional

def gfactor_SENSE_PMR(R_ref: callable,
                      R_acc: callable,
                      ksp_ref: torch.Tensor,
                      ksp_acc: torch.Tensor,
                      noise_var: Optional[float] = 1e-2,
                      n_replicas: Optional[int] = 100,
                      verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Calculates the g-factor map of a SENSE reconstruction using Psuedo Multiple Replica method.
    
    Parameters:
    ----------
    R_ref : callable
        Reference linear reconsruction operator mapping from k-space to image domain
    R_acc : callable
        Accelerated linear reconsruction operator mapping from k-space to image domain
    ksp_ref : torch.Tensor
        Reference k-space data (can be zeros)
    ksp_acc : torch.Tensor
        Accelerated k-space data (can be zeros)
    noise_var : float
        Variance of k-space pseudo-noise
    n_replicas : int
        Number of replicas to use.
    verbose : bool
        Toggles progress bar
        
    Returns:
    --------
    gfactor : torch.Tensor
        g-factor map with same size as image.
    """
    
    var_ref = calc_variance_PMR(R_ref, ksp_ref, noise_var, n_replicas, verbose)
    var_acc = calc_variance_PMR(R_acc, ksp_acc, noise_var, n_replicas, verbose)
    gfactor = (var_acc / var_ref).sqrt()
    return gfactor

def gfactor_SENSE_diag(AHA_inv_ref: callable,
                       AHA_inv_acc: callable,
                       inp_example: torch.Tensor,
                       n_replicas: Optional[int] = 100,
                       rnd_vec_type: Optional[str] = None,
                       verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Calculates the g-factor map of a SENSE reconstruction using diagonal estimation method.
    
    Parameters:
    ----------
    AHA_inv_ref : callable
        Reference inverse gram operator
    AHA_inv_acc : callable
        Accelerated inverse gram operator
    inp_example : torch.Tensor
        Example input to AHA_inv_ref/AHA_inv_acc, helps in determining the shape, device, and dtype of the input.
    n_replicas : int
        Number of replicas to use.
    rnd_vec_type : str
        Type of random vectors to use. Can be 'real' or 'complex'. Defaults to datatype of input.
    verbose : bool
        Toggles progress bar
    
    Returns:
    --------
    gfactor : torch.Tensor
        g-factor map with same size as image.    
    """
    
    diag_ref = diagonal_estimator(AHA_inv_ref, inp_example, n_replicas, rnd_vec_type, verbose)
    diag_acc = diagonal_estimator(AHA_inv_acc, inp_example, n_replicas, rnd_vec_type, verbose)
    gfactor = (diag_acc / diag_ref).sqrt()
    return gfactor

def calc_variance_PMR(R: callable,
                      ksp: torch.Tensor,
                      noise_var: Optional[float] = 1e-2,
                      n_replicas: Optional[int] = 100,
                      verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Comptutes variance map of reconstruction R(noise) using Psuedo Mulitple Replica method:
    
    var = \sum_n |R(noise_n)|^2 / (N * noise_var)
    
    where noise_n is i.i.d. Gaussian noise with variance noise_var.
    
    Robson PM, et. al. Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions. Magn Reson Med. 2008 Oct;60(4):895-907. doi: 10.1002/mrm.21728. PMID: 18816810; PMCID: PMC2838249.
    
    Parameters:
    ----------
    R : callable
        Linear reconsruction operator mapping from k-space to image domain.
    ksp : torch.Tensor
        k-space data (can be zeros)
    noise_var : float
        Variance of k-space pseudo-noise 
    n_replicas : int
        Number of replicas to use.
    verbose : bool
        Toggles progress bar
        
    Returns: 
    --------
    var : torch.Tensor
        variance map with same size as image.
    """
    var = None
    for n in tqdm(range(n_replicas), 'PMR Loop', disable=not verbose):
        noise = (noise_var ** 0.5) * torch.randn_like(ksp * 0)
        recon = R(ksp + noise)
        var_n = recon.abs() ** 2 / (n_replicas * noise_var)
        if var is None:
            var = var_n
        else:
            var += var_n
    return var

def diagonal_estimator(M: callable,
                      inp_example: torch.Tensor,
                      n_replicas: Optional[int] = 100,
                      rnd_vec_type: Optional[str] = None,
                      verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Estimates the diagoal elements of some matrix operator M using a modified Hutchinson's method.
    
    Parameters:
    ----------
    M : callable
        Square linear operator
    inp_example : torch.Tensor
        Example input to M, helps in determining the shape, device, and dtype of the input.
    n_replicas : int
        Number of replicas to use.
    rnd_vec_type : str
        Type of random vectors to use. Can be 'real' or 'complex'. Defaults to datatype of input.
    verbose : bool
        Toggles progress bar
        
    Returns:
    --------
    diag : torch.Tensor
        Estimated diagonal elements of M.
    
    Dharangutte, Prathamesh, and Christopher Musco. "A tight analysis of hutchinson's diagonal estimator." Symposium on Simplicity in Algorithms (SOSA). Society for Industrial and Applied Mathematics, 2023.
    """
    # Get constants from example input 
    idtype = inp_example.dtype
    idevice = inp_example.device
    ishape = inp_example.shape
    
    # Random vector generators
    rnd_vec_comp = lambda : torch.exp(1j * 2 * torch.pi * torch.rand(ishape, device=idevice)).type(idtype)
    rnd_vec_real = lambda : 2 * torch.randint(0, 2, ishape, device=idevice).type(idtype) - 1
    
    # Function to generate random vectors
    if 'real' in rnd_vec_type:
        rnd_vec = rnd_vec_real
    else:
        rnd_vec = rnd_vec_comp
        
    # Estimate diagonal
    diag = torch.zeros_like(inp_example)
    for n in tqdm(range(n_replicas), 'Diagonal Estimator Loop', disable=not verbose):
        v = rnd_vec()
        Mv = M(v)
        diag += (v.conj() * Mv) / n_replicas
    return diag.real

