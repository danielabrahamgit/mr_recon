import torch.fft as fft

def batch_iterator(total, batch_size):
    assert total > 0, f'batch_iterator called with {total} elements'
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])

def sp_fft(x, dim=None, oshape=None):
    """Matches Sigpy's fft, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.fftn(x, s=oshape, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x

def sp_ifft(x, dim=None, oshape=None):
    """Matches Sigpy's fft adjoint, but in torch"""
    x = fft.ifftshift(x, dim=dim)
    x = fft.ifftn(x, s=oshape, dim=dim, norm='ortho')
    x = fft.fftshift(x, dim=dim)
    return x