"""
Typing hints for the C++ functions
"""

import numpy as np


# ------------------------ MATH --------------------------------

def mvn_logpdf(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """
    Computes the log PDF of a Multivariate Normal

    Parameters:
    - x: Point where the lodpdf should be computed
    - mean: Mean of the Gaussian
    - covariance: Covariance matrix of the Gaussian
    """
    ...

def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """
    Computes the squared Mahalanobis distance

    Parameters:
    - x: Point to which the distance should be computed
    - mean: Mean of the Gaussian
    - covariance: Covariance matrix of the Gaussian
    """
    ...
