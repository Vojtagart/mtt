import numpy as np
from numpy.typing import DTypeLike


class Gater:
    """
    Structure used to perform gating over a fixed set of measurements
    """

    def __init__(self, dim: int, dtype: DTypeLike = np.float64) -> None:
        """
        Initializes empty Gater

        Parameters:
        - dim: Dimension of the measurements
        - dtype: Data type
        """
        ...

    def set_measurements(self, measurements: np.ndarray) -> None:
        """
        Sets a new set of measurements
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM).
        """
        ...

    def gate(self, mean: np.ndarray, cov: np.ndarray, threshold: float) -> np.ndarray:
        """
        Performs gating with respect to given Gaussian parameters.
        Returns the indices of measurements whose squared Mahalanobis distance 
        is within the given threshold.
        
        Parameters:
        - mean: Mean of the Gaussian. Shape: (DIM,) or (DIM, 1)
        - cov: Covariance of the Gaussian. Shape: (DIM, DIM)
        - threshold: Maximum squared Mahalanobis distance

        Returns
        - 1D array of indices of measurements that fall inside the gate
        """
        ...