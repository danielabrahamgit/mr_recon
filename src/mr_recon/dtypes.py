import torch
import numpy as np

real_dtype = torch.float64
complex_dtype = torch.complex128
if real_dtype == torch.float64:
    np_real_dtype = np.float64
elif real_dtype == torch.float32:
    np_real_dtype = np.float32
else:
    raise ValueError(f'Unsupported real dtype: {real_dtype}')

if complex_dtype == torch.complex128:
    np_complex_dtype = np.complex128
elif complex_dtype == torch.complex64:
    np_complex_dtype = np.complex64
else:
    raise ValueError(f'Unsupported complex dtype: {complex_dtype}')
