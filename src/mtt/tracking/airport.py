import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from matplotlib.patches import Ellipse
from typing import Optional, List, Tuple
from ..utils.utils import get_cov_ellipse


class Airport:
    """
    Represents a spatial birth location (airport) for target generation

    Models a Gaussian birth component N(x; mu, P) where new
    trajectories originate. Contains the initial state, uncertainty covariance,
    and a relative weight (e.g., expected number of births per scan)
    """

    def __init__(self, x: ArrayLike, P: ArrayLike, weight: float, dtype: DTypeLike = np.float64):
        """
        Initializes a new Airport birth model

        Parameters:
        - x: Initial state vector of planes starting from this airport
        - P: Initial covariance matrix representing spatial/kinematic uncertainty
        - weight: Statistical weight of this airport (birth rate)
        - dtype: NumPy data type

        Raises:
        - ValueError: If P is not a square matrix
        - ValueError: If the length of x does not match the dimension of P
        """
        self.x = np.asarray(x, dtype=dtype).ravel().copy()
        self.P = np.asarray(P, dtype=dtype).copy()

        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Airport P must be a square covariance matrix")
        if self.x.shape[0] != self.P.shape[0]:
            raise ValueError("Length of state vector x must match dimensions of P")
        
        self.P = 0.5 * (self.P + self.P.T)
        self.w = float(weight)

    def plot(self, ellipse_art: Ellipse, n_std: float = 3., pos_idx: Tuple[int, int] = (0, 1)):
        """
        Updates a Matplotlib Ellipse artist to represent the spatial covariance

        Parameters:
        - ellipse_art: The Matplotlib `Ellipse` patch to be updated
        - n_std: Number of standard deviations the ellipse should encompass
        - pos_idx: Tuple of two integers specifying which state vector indices
          correspond to the 2D (x, y) position. Defaults to (0, 1)
        """
        pos_idx = list(pos_idx)
        cov_ix = np.ix_(pos_idx, pos_idx)
        x, P = self.x[pos_idx], self.P[cov_ix]

        w, h, angle = get_cov_ellipse(P, n_std=n_std)
        ellipse_art.set_width(w)
        ellipse_art.set_height(h)
        ellipse_art.set_angle(angle)
        ellipse_art.set_center((x[0], x[1]))

def gen_edge_airports(n: int, radius: float, P: ArrayLike, weight: float, n_std: float = 1., loc: Optional[ArrayLike] = None) -> List[Airport]:
    """
    Generates equally separated airports along the edge of the radar FOV

    Calculates the spatial offset based on the major axis of the covariance 
    ellipse to ensure the `n_std` boundary of the airport fits entirely within 
    the radar radius

    Parameters:
    - n: Number of airports to generate
    - radius: Radius of the radar FOV
    - P: Covariance matrix for each airport
    - weight: Weight assigned to each airport
    - n_std: Standard deviations from the center to keep inside the FOV edge
    - loc: (x, y) location of the radar (defaults to origin)

    Returns:
    - A list containing the instantiated Airport objects
    """
    if n <= 0:
        return []

    P = np.asarray(P)
    dtype = P.dtype

    if loc is None:
        loc = np.array([0.0, 0.0])
    else:
        loc = np.asarray(loc).ravel()

    major, _, _ = get_cov_ellipse(P[:2,:2], n_std)
    RADIUS = radius - major / 2.
    airports = []
    rad = 2. * np.pi / n

    for i in range(n):
        theta = rad * i
        state = np.zeros(P.shape[0], dtype=dtype)
        state[0] = np.cos(theta) * RADIUS + loc[0]
        state[1] = np.sin(theta) * RADIUS + loc[1]
        airports.append(Airport(state, P, weight, dtype=dtype))

    return airports
