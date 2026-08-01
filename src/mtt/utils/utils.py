import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from typing import Optional, Tuple, Union
from matplotlib.patches import Polygon, Ellipse


def get_rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    """
    Obtain a numpy random number generator.

    Prefers an existing `np.random.Generator` if provided; otherwise constructs 
    a new one from `seed` using `np.random.default_rng`

    Parameters:
    - rng: Explicit generator to use (preferred)
    - seed: Integer seed for default_rng if rng is None

    Returns:
    - The supplied or newly created generator
    """
    if rng is None:
        return np.random.default_rng(seed=seed)
    return rng

def sample_circle(radius: float, center: ArrayLike = (0., 0.), size: int = 1, rng: Optional[np.random.Generator] = None,
                  dtype: DTypeLike = np.float64, inner_radius: float = 0.) -> np.ndarray:
    """
    Samples points uniformly from a circle or annulus

    Parameters:
    - radius: Outer radius of the circle
    - center: (x, y) coordinates of the center
    - size: Number of points to sample
    - rng: Random number generator
    - dtype: Numpy data type for the returned array
    - inner_radius: Inner radius to create a ringlet (default 0)

    Returns:
    - Array of sampled points of shape (size, 2)

    Raises:
    - ValueError: If inner_radius is outside [0, radius]
    - ValueError: If the center array does not have exactly 2 elements
    """

    if not (0 <= inner_radius <= radius):
        raise ValueError("Inner_radius must be in range [0, radius]")

    center = np.asarray(center, dtype=dtype).ravel()
    if center.size != 2:
        raise ValueError("Center must have length 2")
    
    rng = get_rng(rng)
    r = rng.random(size) * (radius**2 - inner_radius**2) + inner_radius**2
    r = np.sqrt(r)
    theta = rng.random(size) * 2.0 * np.pi

    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return np.column_stack((x, y)).astype(dtype)

def get_cov_ellipse(cov: ArrayLike, n_std: float = 1.0, ell_params: Optional[dict] = None, pos: Tuple[float, float] = (0., 0.)) -> Union[Tuple[float, float, float], Ellipse]:
    """
    Converts a 2x2 covariance matrix to geometric ellipse parameters

    Parameters:
    - cov: 2x2 covariance matrix
    - n_std: Number of standard deviations the ellipse should represent
    - ell_params: Dictionary of kwargs for `matplotlib.patches.Ellipse`
    - pos: Center position for the `Ellipse` object

    Returns:
    - Tuple[float, float, float]: (major_axis, minor_axis, angle_in_degrees) if ell_params is None
      Axes represent diameters, not radii
    - Ellipse: A `matplotlib.patches.Ellipse` instance if ell_params is provided

    Raises:
    - ValueError: If cov is not a 2x2 matrix.
    """
    cov = np.asarray(cov)
    if cov.shape != (2, 2):
        raise ValueError("Covariance matrix isn't of dim (2,2)")

    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    vals = np.sqrt(vals) * n_std * 2
    angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    if ell_params is not None:
        return Ellipse(pos, width=vals[1], height=vals[0], angle=angle, **ell_params)
    return vals[1], vals[0], angle

def make_arrow(pos: ArrayLike, dir: ArrayLike, scale: float = 1., points: bool = False) -> Union[np.ndarray, Polygon]:
    """
    Creates a 2D arrow-shaped polygon oriented along a direction vector

    Parameters:
    - pos: 2-element sequence specifying the centroid
    - dir: 2-element direction vector. Magnitude is ignored
    - scale: Scalar multiplier for the base coordinates
    - points: If True, returns raw vertices. Otherwise, returns a closed Polygon

    Returns:
    - np.ndarray: Float array of shape (4,2) with vertex coordinates (if points=True).
    - Polygon: Matplotlib polygon object (if points=False).
    """
    pos = np.asarray(pos, dtype=float).ravel()
    dir = np.asarray(dir, dtype=float).ravel()

    body = np.array([[1, 0],[-1, 1],[-0.5, 0],[-1, -1]], dtype=float) * float(scale)
    body -= body.mean(axis=0, keepdims=True)

    theta = np.arctan2(dir[1], dir[0])
    # theta = np.arctan2(dir[0], dir[1])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    body =  body @ R.T + pos
    
    if points:
        return body
    return Polygon(body, closed=True)
