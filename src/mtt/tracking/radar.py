import numpy as np
from numpy.typing import DTypeLike
from numpy.random import SeedSequence, default_rng
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from ..utils.utils import sample_circle
from .trajectory import Trajectory


class Radar:
    """
    Simple 2D radar/clutter simulator

    Simulates detections from a set of `Trajectory` objects and 
    Poisson-distributed spatial clutter within a circular field-of-view (FOV)
    Currently supports only 2D position measurements
    """

    def __init__(self, lambd: float, det_prob: float = 1.0, radius: float = 1000., loc: np.ndarray = None,
                 gate_by_rad: bool = True, seed: Optional[int] = None, dtype: DTypeLike = np.float64):
        """
        Initializes the Radar simulator

        Parameters:
        - lambd: Spatial clutter intensity (points per unit area)
        - det_prob: Probability of detection for each target [0, 1]
        - radius: Radial range of the radar FOV
        - loc: 2-element array-like specifying radar center (defaults to [0,0])
        - gate_by_rad: If True, targets outside `radius` are strictly missed
        - seed: Integer seed for reproducible per-time RNGs
        - dtype: NumPy dtype used for clutter and detection arrays

        Raises:
        - ValueError: If `det_prob`, `radius`, or `lambd` are outside valid ranges
        - ValueError: If `loc` does not have exactly 2 elements
        """

        if not (0. <= det_prob <= 1.):
            raise ValueError(f"det_prob must be in range [0, 1], got {det_prob}")
        if radius <= 0.:
            raise ValueError(f"expected positive radius, got {radius}")
        if lambd < 0.:
            raise ValueError(f"lambd (spatial density) must be non-negative, got {lambd}")
        
        if loc is None:
            loc = np.array([0., 0.])
        else:
            loc = np.asarray(loc, dtype=float).ravel()

        if loc.size != 2:
            raise ValueError(f"Loc has to consists of 2 elements")

        self.exp_clut = (np.pi * radius**2) * lambd
        self.det_prob = float(det_prob)
        self.radius = float(radius)
        self.loc = loc.copy()
        self.gate_by_rad = gate_by_rad
        self.seed = int(seed) if seed is not None else int(default_rng().integers(0, 2**31 - 1))
        self.dtype = dtype

        self.dets = np.empty((0, 2), dtype=dtype)
        self.clut = np.empty((0, 2), dtype=dtype)

    def detect(self, time: int, trajs: List[Trajectory], change_mask: bool = True) -> np.ndarray:
        """
        Performs one detection sweep at a given time step

        Generates clutter and tests active trajectories for detection based on 
        `det_prob` and spatial boundaries. Optionally updates trajectory masks

        Parameters:
        - time: Time index at which detections are generated
        - trajs: List of `Trajectory` instances active at `time`
        - change_mask: If True, updates the trajectory's internal detection mask

        Returns:
        - Array of shape (N_total, 2) containing concatenated clutter 
          and true detections. Shape is (0, 2) if no points exist
        """

        rng = self._rng_for_time(time)
        n_clut = rng.poisson(self.exp_clut)
        if n_clut == 0:
            self.clut = np.empty((0, 2), dtype=self.dtype)
        else:
            self.clut = sample_circle(self.radius, center=self.loc, size=n_clut, rng=rng, dtype=self.dtype)
        
        dets = []
        for traj in trajs:
            idx = traj.time_to_index(time)
            if idx is None:
                continue
            meas = traj.meas_at(time, ignore_mask=True)
            meas_pos = traj.model.get_meas_pos(meas)[:2]

            is_detected = rng.random() < self.det_prob
            if is_detected and self.gate_by_rad:
                dist = np.linalg.norm(meas_pos - self.loc)
                if dist > self.radius:
                    is_detected = False
            if is_detected:
                dets.append(meas_pos)  
            if change_mask:
                traj.mask[idx] = is_detected

        self.dets = np.asarray(dets, dtype=self.dtype) if len(dets) > 0 else np.empty((0, 2), dtype=self.dtype)
        return np.concatenate([self.clut, self.dets], axis=0)

    def _rng_for_time(self, time: int) -> np.random.Generator:
        """
        Creates a deterministic RNG derived from the base seed and time index

        Parameters:
        - time: Integer time index

        Returns:
        - np.random.Generator: Seeded uniquely and deterministically for the given time
        """
        ss = SeedSequence([self.seed, int(time)])
        return default_rng(ss)

    def plot(self, clut_art: Optional[PathCollection] = None, det_art: Optional[PathCollection] = None, relabel: bool = True):
        """
        Updates Matplotlib scatter artists with the latest `detect()` results

        Parameters:
        - clut_art: Scatter artist for clutter points
        - det_art: Scatter artist for true detections
        - relabel: If True, updates artist labels with current point counts
        """

        if clut_art is not None:
            clut_art.set_offsets(self.clut)
            if relabel:
                clut_art.set_label(f"clutter ({self.clut.shape[0]})")

        if det_art is not None:
            det_art.set_offsets(self.dets)
            if relabel:
                det_art.set_label(f"cur. dets. ({self.dets.shape[0]})")
        
    def plot_base(self, ax: plt.Axes, size: Optional[float] = None, circles: int = 2) ->Tuple[Rectangle, List[Circle]]:
        """
        Draws the static radar visualization on an Axes

        Draws a centered marker and concentric dashed range rings

        Parameters:
        - ax: Matplotlib Axes object
        - size: Side length of the center square (defaults to 5% of radius)
        - circles: Number of concentric range rings to draw

        Returns:
        - Tuple[Rectangle, List[Circle]]: The center patch and the range rings
        """
        if size is None:
            size = self.radius * 0.05

        cx, cy = float(self.loc[0]), float(self.loc[1])
        rect = Rectangle((cx - size/2, cy - size/2), size, size, facecolor='none', edgecolor='black')
        ax.add_patch(rect)
        circles_lst = []
        for i in range(1, circles + 1):
            x = i / circles
            c = Circle((cx, cy), self.radius * x, facecolor='none', edgecolor='gray', ls='--')
            ax.add_patch(c)
            circles_lst.append(c)
        return rect, circles_lst
