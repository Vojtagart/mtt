from .. import _core
import numpy as np
from numpy.typing import DTypeLike
from .utils import _get_binding_class


def Gater(dim: int = None, dtype: DTypeLike = np.float64):
    """
    Initializes a Gater with given dimension

    Parameters:
    - dim: Dimension of the measurements (None if dynamic)
    - dtype: Data type
    """

    if dim <= 0:
        raise ValueError("Dimension must be > 0")
    cls = _get_binding_class("_Gater", dtype, (dim,))
    return cls(dim)
