"""
Inspired by the sigpy library, this package is a collection of tools for MRI reconstruction.
"""
import sys
from ._func import block as _block
from ._func import pad as _pad
from ._func import indexing as _indexing
sys.modules[__name__ + ".pad"] = _pad
sys.modules[__name__ + ".indexing"] = _indexing
sys.modules[__name__ + ".block"] = _block
