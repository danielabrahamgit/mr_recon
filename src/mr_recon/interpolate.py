import torch
import numpy as np
import gpytorch

from tqdm import tqdm
from typing import Optional
from mr_recon.algs import lin_solve

class BatchedOutputGPModel(gpytorch.models.ExactGP):
    """
    Defines a GP model capable of modeling multiple outputs (tasks).
    """
    def __init__(self, 
                 X: torch.Tensor, 
                 Y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods.Likelihood):
        """
        Args:
        -----
        X : torch.Tensor
            Input data of shape (N, d) where N is the number of samples and d is the dimensionality.
        Y : torch.Tensor
            Output data of shape (N, p) where N is the number of samples and p is the number of tasks.
        likelihood : gpytorch.likelihoods.Likelihood
            Likelihood function for the GP model.
        """
        super(BatchedOutputGPModel, self).__init__(X, Y, likelihood)
        self.N = X.shape[0]
        self.p = Y.shape[-1]
        self.d = X.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=(self.p,))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.p])),
            batch_shape=torch.Size([self.p])
        )

    def predict(self,
                x: torch.Tensor,) -> torch.Tensor:
        """
        Predict the output for the given input using the GP model.

        Args:
        -----
        x : torch.Tensor
            Input data of shape (..., d) 
        
        Returns:
        -------
        torch.Tensor
            Predicted output of shape (..., p) where p is the number of tasks.
        """
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_flt = x.reshape((-1, self.d))
            predictions = self.likelihood(self(x_flt))
            return predictions.mean.reshape((*x.shape[:-1], self.p))
            # y_flt = self.mean_module(x.reshape((-1, self.d)))
            # return y_flt.T.reshape((*x.shape[:-1], self.p))
    
    def forward(self, 
                x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """
        Returns a MultitaskMultivariateNormal distribution over the outputs.

        Args:
        -----
        x : torch.Tensor
            Input data of shape (M, d) where d is the dimensionality.

        Returns:
        --------
            gpytorch.distributions.Distribution: _description_
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )
    
def gp_model(X: torch.Tensor, 
             Y: torch.Tensor) -> BatchedOutputGPModel:
    """
    Create a GP model for the given input and output data.

    Args:
    -----
    X : torch.Tensor
        Input data of shape (N, d) where N is the number of samples and d is the dimensionality.
    Y : torch.Tensor
        Output data of shape (N, p) where N is the number of samples and p is the number of tasks.

    Returns:
    -------
    MultitaskGPModel
        The GP model.
    """
    ll = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y.shape[-1])
    return BatchedOutputGPModel(X, Y, ll)

def optimize_hyper(model: BatchedOutputGPModel, 
                   X: torch.Tensor, 
                   Y: torch.Tensor, 
                   num_iter: int = 100,
                   lr: Optional[float] = 1e-1) -> BatchedOutputGPModel:
    """
    Optimize the hyperparameters of the GP model.

    Args:
    -----
    model : MultitaskGPModel
        The GP model.
    X : torch.Tensor
        Input data of shape (N, d) where N is the number of samples and d is the dimensionality.
    Y : torch.Tensor
        Output data of shape (N, p) where N is the number of samples and p is the number of tasks.
    num_iter : int
        Number of iterations for optimization.

    Returns:
    -------
    MultitaskGPModel
        The optimized GP model.
    """
    # Prep model for training
    model.train()
    model.likelihood.train()

    # Optimize hyperparams over log likelihood loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Main loop
    pbar = tqdm(range(num_iter), 'Optimizing hyperparameters')
    for i in pbar:
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=f"{loss:.4f}")

    return model



def apply_interp_kernel(xs: torch.Tensor,
                        x: torch.Tensor,
                        kern_weights: torch.Tensor,
                        kern_func: callable,
                        batch_size: Optional[int] = None,
                        verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Perform radial basis function interpolation.

    Parameters:
    -----------
    xs : (torch.Tensor)
        The observed coordinates with shape (K, d)
    x : (torch.Tensor)
        The coordinates to interpolate at with shape (..., d)
    kern_weights : (torch.Tensor)
        The weights for the kernel with shape (K, n)
    kern_func : (callable)
        The kernel function to use for interpolation
        maps (x1, x2) -> scalar 'distance'
    
    Returns:
    --------
    y : (torch.Tensor)
        The interpolated tensor with shape (..., n)
    """
    # Consts
    arb_shape = x.shape[:-1]
    N = np.prod(arb_shape)
    K = xs.shape[0]
    d = xs.shape[1]
    n = kern_weights.shape[1]
    assert K == kern_weights.shape[0]
    
    # Interpolate over batches
    x_flt = x.reshape((-1, d))
    y_flt = torch.zeros((N, n), device=x.device, dtype=kern_weights.dtype)
    batch_size = N if batch_size is None else batch_size
    for n1 in tqdm(range(0, N, batch_size), 'Applying Kernels', disable=not verbose):
        n2 = min(n1 + batch_size, N)
        
        # apply weights
        data_matrix = kern_func(x_flt[n1:n2, None], xs[None, :]).type(kern_weights.dtype) # N K
        y_flt[n1:n2] = data_matrix @ kern_weights # N n
    
    return y_flt.reshape((*arb_shape, n))

def get_interp_kernel(xs: torch.Tensor,
                      ys: torch.Tensor,
                      kern_type: Optional[str] = 'rbf',
                      kern_param: Optional[float] = 1.0,
                      lamda: Optional[float] = 0.0,
                      auto_cal: Optional[bool] = False) -> tuple[torch.Tensor, callable]:
    """
    Get radial basis interpolation function and kernel weights

    Parameters:
    -----------
    xs : (torch.Tensor)
        The observed coordinates with shape (K, d)
    ys : (torch.Tensor)
        The observed values with shape (K, n)
    x : (torch.Tensor)
        The coordinates to interpolate at with shape (..., d)
    kern_type : (Optional[str])
        The distance kernel to use for interpolation, options are:
            - 'rbf': radial basis function (default), param is sigma
    kern_param : (Optional[float])
        See above
    lamda : (Optional[float])
        Regularization parameter for the kernel weights
    auto_cal : (Optional[bool])
        If True, will automatically calculate the kernel parameter based on the
        distance between the points. This is not recommended for large datasets.
    
    Returns:
    --------
    y : (torch.Tensor)
        The interpolated tensor with shape (..., n)
    """
    # Consts
    K = xs.shape[0]
    d = xs.shape[1]
    n = ys.shape[1]
    assert xs.shape[0] == ys.shape[0], "xs and ys must have the same first dimension"
    
    # Get kernel function
    known_kernels = ['rbf', 'rbf_l1']
    if kern_type not in known_kernels:
        raise ValueError(f"Unknown kernel type: {kern_type}, must be one of {known_kernels}")
    if kern_type == 'rbf':
        sigma = kern_param
        def kern_func(x1, x2):
            return torch.exp(-(torch.linalg.norm(x1 - x2, dim=-1) / sigma) ** 2)
    elif kern_type == 'rbf_l1':
        sigma = kern_param
        def kern_func(x1, x2):
            return torch.exp(-torch.linalg.norm(x1 - x2, dim=-1) / sigma)
    
    # Build kernel matrix 
    A = kern_func(xs[:, None], xs[None, :]).type(ys.dtype)  # K M
    
    # Build target matrix
    B = ys
    
    # Solve
    kern_weights = lin_solve(A.H @ A, A.H @ B, lamda=lamda, solver='solve') # K n
    B_est = A @ kern_weights
    print(f'{100*(B - B_est).norm() / B.norm():.3f} pcnt error on training data')
    
    return kern_weights, kern_func
