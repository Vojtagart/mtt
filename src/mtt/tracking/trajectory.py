import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from typing import Optional, Union
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from ..utils.utils import make_arrow, get_rng
from ..utils.numpy_vector import NumpyVector
from .model import Model


class Trajectory:
    """
    Simulated discrete-time trajectory for a linear Gaussian state-space model

    Generates sequential states and measurements based on an initial state and 
    the provided linear transition/measurement matrices with additive white 
    Gaussian noise. Built-in plotting assumes 2D position + velocity
    """

    def __init__(self, model: Model, x0: ArrayLike, ndat: int, start_time: int = 0, initial_state_noise: bool = False,
                 seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        """
        Constructs a trajectory realization

        Parameters:
        - model: `Model` instance defining (A, H, Q, R, dtype, dim)
        - x0: Initial state vector of shape (s,), where s = model.state_dim
        - ndat: Number of discrete time steps to generate (> 0)
        - start_time: Starting time index (default: 0)
        - initial_state_noise: If True, adds process noise to the initial state
        - seed: Integer seed for RNG if `rng` is None
        - rng: Explicit `np.random.Generator` (preferred over seed)

        Raises:
        - ValueError: If `ndat <= 0` or If `x0` length does not match `model.state_dim`
        """
        dtype = model.dtype
        ndat = int(ndat)
        x0 = np.asarray(x0, dtype=dtype).ravel()

        if ndat <= 0:
            raise ValueError(f"Expected positive ndat, got {ndat}")
        if model.state_dim != x0.shape[0]:
            raise ValueError("x0 must be have the same length as model.state_dim")
        
        self.model = model
        self.start = int(start_time)
        self.rng = get_rng(seed=seed, rng=rng)
        self._X = NumpyVector(model.state_dim, dtype, np.zeros((ndat, model.state_dim), dtype=dtype))
        self._Y = NumpyVector(model.meas_dim, dtype, np.zeros((ndat, model.meas_dim), dtype=dtype))
        self._mask = NumpyVector(1, bool, np.ones((ndat, 1), dtype=bool))
        self._generate(self.X, self.Y, x0, initial_state_noise=initial_state_noise)

    @property
    def size(self) -> int:
        """Number of data points in the trajectory"""
        return self._X.size

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector"""
        return self.model.state_dim

    @property
    def meas_dim(self) -> int:
        """Dimension of the measurement vector"""
        return self.model.meas_dim
    
    @property
    def dtype(self) -> DTypeLike:
        """Numpy dtype of the trajectory arrays"""
        return self.model.dtype
    
    @property
    def X(self) -> np.ndarray:
        """State sequence array of shape (size, state_dim)"""
        return self._X.data
    
    @property
    def Y(self) -> np.ndarray:
        """Measurement sequence array of shape (size, meas_dim)"""
        return self._Y.data
    
    @property
    def mask(self) -> np.ndarray:
        """1D boolean mask of valid detections of length `size`"""
        return self._mask.data[:, 0]
    
    @property
    def end(self) -> int:
        """Final global time index (exclusive)"""
        return self.start + self.size
    
    def __lt__(self, other: "Trajectory"):
        """Compares trajectories based on their end time"""
        return self.end < other.end
    
    def time_to_index(self, time: int) -> Union[int, None]:
        """
        Converts a global time index to the internal array index

        Negative times index backwards relative to the current end

        Parameters:
        - time: Integer global time

        Returns:
        - int: Array index in [0, size-1] if within bounds
        - None: If the time is out of range
        """
        if time < 0:
            time = self.start + self.size + int(time)

        idx = int(time) - self.start
        if idx < 0 or idx >= self.size:
            idx = None
        return idx
    
    def state_at(self, time: int, copy: bool = False) -> np.ndarray:
        """
        Retrieves the true state vector at a specific global time

        Parameters:
        - time: Global time index
        - copy: If True, returns a copy; otherwise returns a view

        Returns:
        - np.ndarray: State vector of shape (state_dim,). Returns an empty array if out of range
        """
        idx = self.time_to_index(time)
        if idx is None:
            return np.empty((self.state_dim, 0), dtype=self.dtype)
        return self.X[idx].copy() if copy else self.X[idx]
    
    def meas_at(self, time: int, copy: bool = False, ignore_mask: bool = False) -> np.ndarray:
        """
        Retrieves the measurement vector at a specific global time

        Parameters:
        - time: Global time index
        - copy: If True, returns a copy; otherwise returns a view
        - ignore_mask: If False, masked-out detections return an empty array

        Returns:
        - np.ndarray: Measurement vector of shape (meas_dim,). Returns an empty array if out of range or masked
        """
        idx = self.time_to_index(time)
        if idx is None or (not ignore_mask and not self.mask[idx]):
            return np.empty((self.meas_dim, 0), dtype=self.dtype)
        return self.Y[idx].copy() if copy else self.Y[idx]
    
    def __len__(self):
        """Number of time steps in the trajectory"""
        return self.size

    def set_time(self, start_time: int):
        """
        Resets the starting global time of the trajectory

        Parameters:
        - start_time: New start time index
        """

        self.start = int(start_time)

    def shift_coords(self, shift: ArrayLike):
        """
        Applies a spatial translation to both state and measurement vectors

        Parameters:
        - shift: Translation vector. Will be applied to the first `len(shift)` coordinates

        Raises:
        - RuntimeError: If model lacks position getters
        - ValueError: If shift length exceeds spatial position dimensions
        """
        shift = np.asarray(shift, dtype=self.dtype).flatten()
        s_dim = shift.shape[0]

        if self.model.state_pos is None or self.model.meas_pos is None:
            raise RuntimeError("Cannot shift a trajectory based on model with no position getters")
        
        s_idx = np.arange(self.state_dim)[self.model.state_pos][:s_dim]
        m_idx = np.arange(self.meas_dim)[self.model.meas_pos][:s_dim]
        
        if s_dim != len(s_idx) or s_dim != len(m_idx):
            raise ValueError("Shift length must equal spatial position dimensions")

        self.X[:, s_idx] += shift
        self.Y[:, m_idx] += shift

    def extend(self, ndat: int):
        """
        Extends the trajectory forward by simulating additional steps

        Parameters:
        - ndat: Number of extra steps to generate (> 0)

        Raises:
        - ValueError: If `ndat <= 0`
        """
        if ndat <= 0:
            raise ValueError(f"ndat must be positve, got {ndat}")

        st = self.size
        end = self.size + ndat
        x0 = self.X[-1].copy()
        self._X.extend(np.zeros((ndat, self.state_dim), dtype=self.dtype))
        self._Y.extend(np.zeros((ndat, self.meas_dim), dtype=self.dtype))
        self._mask.extend(np.ones((ndat, 1), dtype=bool))
        self._generate(self.X[st:end], self.Y[st:end], x0, initial_state_noise=True, keep_x0=False)

    def shrink_to_fit(self):
        """Trims unused container capacity to match the actual trajectory size"""
        self._X.shrink_to_fit()
        self._Y.shrink_to_fit()
        self._mask.shrink_to_fit()

    def resize(self, nsize):
        """Resizes the trajectory to new size"""
        nsize = max(0, int(nsize))
        if nsize > self.size:
            self.extend(nsize - self.size)
        elif nsize < self.size:
            self._X.resize(nsize)
            self._Y.resize(nsize)
            self._mask.resize(nsize)

    def plot(self, time: int, arrow_scale: float = 1., x_line: Optional[Line2D] = None,
             y_line: Optional[PathCollection] = None, arrow: Optional[Polygon] = None, mask: Optional[np.ndarray] = None,
             pregen_traj: bool = False, pregen_meas: bool = False):
        """
        Updates matplotlib artists to display the trajectory up to a given time

        Parameters:
        - time: Global time index to display
        - arrow_scale: Scale factor for the velocity direction arrow or static pentagon
        - x_line: `Line2D` object for the state path
        - y_line: `PathCollection` object for measurement scatter points
        - arrow: `Polygon` object for the current velocity vector
        - mask: Optional custom boolean mask. If None, uses internal mask
        - pregen_traj: If True, draws the entire generated trajectory regardless of time
        - pregen_meas: If True, draws all generated measurements regardless of time
        
        Raises:
        - ValueError: If provided mask is shorter than the required index range
        """
        idx = self.time_to_index(time)

        if idx is None:
            if x_line is not None:
                x_line.set_data([], [])
            if y_line is not None:
                y_line.set_offsets(np.empty((0, 2)))
            if arrow is not None:
                arrow.set_xy(np.zeros((4, 2)))
            return

        if x_line is not None:
            idx_ = self.size - 1 if pregen_traj else idx
            pos_x = self.model.get_pos(self.X[:idx_+1])
            if pos_x is not None and pos_x.shape[1] >= 2:
                x_line.set_data(pos_x[:, 0], pos_x[:, 1])

        if y_line is not None:
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape[0] < idx + 1:
                    raise ValueError("mask is shorter than required length")
            else:
                mask = self.mask
            idx_ = self.size - 1 if pregen_meas else idx
            ys = self.Y[:idx_+1][mask[:idx_+1]]
            
            if ys.size == 0:
                y_line.set_offsets(np.empty((0, 2)))
            else:
                meas_pos = self.model.get_meas_pos(ys)
                if meas_pos is not None and meas_pos.shape[1] >= 2:
                    y_line.set_offsets(meas_pos[:, :2])

        if arrow is not None:
            state = self.X[idx]
            pos = self.model.get_pos(state)
            vel = self.model.get_vel(state)
            
            if pos is not None and pos.size >= 2:
                if vel is not None and vel.size >= 2:
                    # plot arrow pointing as per the velocity vector
                    arrow.set_xy(make_arrow(pos[:2], vel[:2], arrow_scale, points=True))
                else:
                    # plot pentagon
                    angles = np.linspace(0, 2 * np.pi, 6)[:-1] + (np.pi / 2.0)
                    pentagon = np.column_stack((np.cos(angles), np.sin(angles))) * arrow_scale + pos[:2]
                    arrow.set_xy(pentagon)

    def _generate(self, X: np.ndarray, Y: np.ndarray, x0: np.ndarray, initial_state_noise: bool = False, keep_x0: bool = True):
        """
        Internal generator for simulating the state and measurement sequences

        Parameters:
        - X: Preallocated state array of shape (n, state_dim)
        - Y: Preallocated measurement array of shape (n, meas_dim)
        - x0: Initial state vector
        - initial_state_noise: If True, adds process noise to step 0
        - keep_x0: If True, X[0] is set directly to x0 (+ noise). Otherwise, propagates once

        Raises:
        - ValueError: If X or Y arrays have incorrect shapes
        """
        n = X.shape[0]
        if (n == 0):
            return
        if X.shape != (n, self.state_dim):
            raise ValueError("X has incorrect shape")
        if Y.shape != (n, self.meas_dim):
            raise ValueError("Y has incorrect shape")
        
        model = self.model

        V = self.rng.multivariate_normal(mean=np.zeros(self.state_dim), cov=model.Q, size=n).astype(self.dtype)
        W = self.rng.multivariate_normal(mean=np.zeros(self.meas_dim), cov=model.R, size=n).astype(self.dtype)

        if not initial_state_noise:
            V[0] = 0.
        if keep_x0:
            X[0] = x0 + V[0]
        else:
            X[0] = model.F @ x0 + V[0]

        for i in range(1, n):
            X[i] = model.F @ X[i - 1] + V[i]
        Y[:] = X @ model.H.T + W
