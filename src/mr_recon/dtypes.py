import torch
import numpy as np

# Define defaults for global variables. Default float32 precision.
real_dtype = torch.float32
complex_dtype = torch.complex64
np_real_dtype = np.float32
np_complex_dtype = np.complex64

SUPPORTED_PRECISIONS = ['float32', 'float64']

def set_precision(precision: str) -> None:
    """
    Update internal precision for datatypes across the library.
    
    Parameters:
        Precision: precision of real-valued data can be 'float32' or 'float64'.
        Numpy, torch, and complex data types are updated accordingly.

    """
    global real_dtype, complex_dtype, np_real_dtype, np_complex_dtype  # Declare global variables

    # Validate the input dtype
    assert precision in SUPPORTED_PRECISIONS, f"Unsupported precision: {precision}. Must be one of {SUPPORTED_PRECISIONS}."

    # Update the global variables
    if precision == 'float32':
        real_dtype = torch.float32
        complex_dtype = torch.complex64
        np_real_dtype = np.float32
        np_complex_dtype = np.complex64
    else:
        real_dtype = torch.float64
        complex_dtype = torch.complex128
        np_real_dtype = np.float64
        np_complex_dtype = np.complex128

def cast_to_dtype(data):
    """
    Cast input data to the current global dtype settings.
    
    Parameters:
        data: A tensor (torch.Tensor) or array (np.ndarray) to be cast.

    Returns:
        The input data cast to the appropriate library dtype.
    """
    # Determine the type of input
    if isinstance(data, torch.Tensor):
        # Determine if it's a real or complex tensor
        target_dtype = complex_dtype if torch.is_complex(data) else real_dtype
        return data.to(dtype=target_dtype)

    elif isinstance(data, np.ndarray):
        # Determine if it's a real or complex NumPy array
        target_dtype = np_complex_dtype if np.iscomplexobj(data) else np_real_dtype
        return data.astype(target_dtype)
    
    elif isinstance(data, (int, float)):
        # Real-valued scalar
        return np_real_dtype(data)
    
    elif isinstance(data, complex):
        # Complex-valued scalar
        return np_complex_dtype(data)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Must be torch.Tensor or np.ndarray.")
