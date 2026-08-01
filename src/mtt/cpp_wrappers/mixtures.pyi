from typing import Tuple
import numpy as np
from numpy.typing import DTypeLike


class GaussianMixture:
    """
    Gaussian Mixture class
    """

    def __init__(self, dim: int, dtype: DTypeLike = np.float64) -> None:
        """
        initalizes a GaussianMixture
        
        Parameters:
        - dim: Dimension of the Gaussians
        - dtype: Data type
        """
        ...

    def push(self, weight: float, mean: np.ndarray, cov: np.ndarray) -> None:
        """
        Adds a new Gaussian component to the mixture

        Parameters:
        - weight: Weight of the component
        - mean: Mean of the Gaussian
        - cov: Covariance of the Gaussian
        """
        ...

    def erase(self, idx: int) -> None:
        """
        Removes a component at the specified index

        Parameters:
        - idx: Index of component to be removed
        """
        ...

    def filter_out(self, min_weight: float) -> None:
        """
        Removes components with a weight strictly below min_weight

        Parameters:
        - min_weight: Minimum weight
        """
        ...

    def scale_weight(self, multiplier: float) -> None:
        """
        Scales the weight of all components by the 
        
        Parameters:
        - multiplier: Scaling parameter
        """
        ...

    def reserve(self, new_cap: int) -> None:
        """
        Reserves memory for the specified number of components

        Parameters:
        - new_cap: Capacity to reserve
        """
        ...

    def clear(self) -> None:
        """
        Clears all components from the mixture
        """
        ...

    def size(self) -> int:
        """
        Returns:
        - Count of components in the mixture
        """
        ...

    def capacity(self) -> int:
        """
        Returns:
        - Capacity of the mixture
        """
        ...

    @property
    def W(self) -> np.ndarray:
        """
        Returns:
        - Weights array of shape (N,)
        """
        ...

    @W.setter
    def W(self, value: np.ndarray) -> None:
        ...

    @property
    def M(self) -> np.ndarray:
        """
        Returns:
        - Means array of shape (N, DIM)
        """
        ...

    @M.setter
    def M(self, value: np.ndarray) -> None:
        ...

    @property
    def C(self) -> np.ndarray:
        """
        Returns:
        - Covariances array of shape (N, DIM, DIM)
        """
        ...

    @C.setter
    def C(self, value: np.ndarray) -> None:
        ...

    def as_gaussian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs moment matching to collapse the mixture into a single Gaussian
        
        Returns:
        - Tuple containing (mean, covariance)
        """
        ...


class MixtureToGaussian:
    """
    Utility to incrementally collapse multiple Gaussian components into a single Gaussian
    via moment matching.
    """

    def __init__(self, dim: int, dtype: DTypeLike = np.float64) -> None:
        """
        initalizes a class MixtureToGaussian:

        Parameters:
        - dim: Dimension of the Gaussians
        - dtype: Data type
        """
        ...

    def add_gauss(self, weight: float, mean: np.ndarray, cov: np.ndarray) -> None:
        """
        Accumulates a Gaussian component for moment matching

        Parameters:
        - weight: Weight of the component
        - mean: Mean of the Gaussian
        - cov: Covariance of the Gaussian
        """
        ...

    def get_gauss(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        - Tuple containing (mean, covariance)
        """
        ...


class MultiBernoulli:
    """
    Multi-Bernoulli representation class
    """

    def __init__(self, dim: int, dtype: DTypeLike = np.float64) -> None:
        """
        initalizes a MultiBernoulli
        
        Parameters:
        - dim: Dimension of the Gaussians
        - dtype: Data type
        """
        ...

    def push(self, exist_prob: float, mean: np.ndarray, cov: np.ndarray) -> None:
        """
        Adds a new Gaussian component to the mixture

        Parameters:
        - exist_prob: Existence probability of the Gaussian
        - mean: Mean of the Gaussian
        - cov: Covariance of the Gaussian
        """
        ...

    def erase(self, idx: int) -> None:
        """
        Removes a component at the specified index

        Parameters:
        - idx: Index of component to be removed
        """
        ...

    def filter_out(self, min_weight: float) -> None:
        """
        Removes components with a weight strictly below min_weight

        Parameters:
        - min_weight: Minimum weight
        """
        ...

    def scale_exist_prob(self, multiplier: float) -> None:
        """
        Scales the existence probability of all components by the multiplier
        
        Parameters:
        - multiplier: Scaling parameter
        """
        ...

    def reserve(self, new_cap: int) -> None:
        """
        Reserves memory for the specified number of components

        Parameters:
        - new_cap: Capacity to reserve
        """
        ...

    def clear(self) -> None:
        """
        Clears all components from the mixture
        """
        ...

    def size(self) -> int:
        """
        Returns:
        - Count of components in the mixture
        """
        ...

    def capacity(self) -> int:
        """
        Returns:
        - Capacity of the mixture
        """
        ...

    @property
    def R(self) -> np.ndarray:
        """
        Returns:
        - Existence probability array of shape (N,)
        """
        ...

    @R.setter
    def R(self, value: np.ndarray) -> None:
        ...

    @property
    def M(self) -> np.ndarray:
        """
        Returns:
        - Means array of shape (N, DIM)
        """
        ...

    @M.setter
    def M(self, value: np.ndarray) -> None:
        ...

    @property
    def C(self) -> np.ndarray:
        """
        Returns:
        - Covariances array of shape (N, DIM, DIM)
        """
        ...

    @C.setter
    def C(self, value: np.ndarray) -> None:
        ...
