from .. import _core
import numpy as np
from numpy.typing import DTypeLike
from .utils import _get_binding_class


def GaussianMixture(dim: int, dtype: DTypeLike = np.float64):
    """
    Initializes a GaussianMixture with given dimension

    Parameters:
    - dim: Dimension of the Gaussians
    - dtype: Data type
    """

    if dim <= 0:
        raise ValueError("Dimension must be > 0")
    cls = _get_binding_class("_GaussianMixture", dtype, (dim,))
    return cls(dim)


def MultiBernoulli(dim: int, dtype: DTypeLike = np.float64):
    """
    Initializes a MultiBernoulli with given dimension

    Parameters:
    - dim: Dimension of the Gaussians
    - dtype: Data type
    """

    if dim <= 0:
        raise ValueError("Dimension must be > 0")
    cls = _get_binding_class("_MultiBernoulli", dtype, (dim,))
    return cls(dim)


def MixtureToGaussian(dim: int, dtype: DTypeLike = np.float64):
    """
    Initializes a MixtureToGaussian

    Parameters:
    - dim: Dimension of the Gaussians
    - dtype: Data type
    """

    if dim <= 0:
        raise ValueError("Dimension must be > 0")
    cls = _get_binding_class("_MixtureToGaussian", dtype, (dim,))
    return cls(dim)