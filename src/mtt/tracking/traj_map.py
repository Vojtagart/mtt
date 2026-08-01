import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, List, Dict, Iterator, Union
from heapq import heappop, heappush
from ..utils.utils import get_rng, sample_circle
from .model import Model
from .trajectory import Trajectory


class TrajMap:
    """
    Random trajectory field generator with a circular radar field-of-view (FOV)

    Produces collections of `Trajectory` objects born according to a Poisson 
    distribution, following a supplied linear-Gaussian `Model`. Trajectories 
    can be seeded from preset initial states ("airports") or generated 
    stochastically inside/outside the FOV

    If the model dimension exceeds 2, higher spatial dimensions are initialized 
    to 0. Maintains a birth index for efficient O(log N) time-wise iteration
    """

    def __init__(self, ndat: int, radius: float, center: ArrayLike = [0, 0], rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = None):
        """
        Creates a TrajMap

        Parameters:
        - ndat: Total number of discrete time steps (trajectory horizon)
        - radius: Radar FOV radius
        - center: 2-element center of the circular FOV (x, y)
        - rng: Optional `np.random.Generator` (preferred over seed)
        - seed: Integer seed for RNG creation
        
        Raises:
        - ValueError: If the center array length is less than 2
        """
        self.size = int(ndat)
        self.trajs: List[Trajectory] = []
        self.rad = float(radius)
        self.center = np.asarray(center, dtype=float).ravel().copy()
        if self.center.size < 2:
            raise ValueError("center must be length >= 2 (x,y,...)")
        self.rng = get_rng(rng=rng, seed=seed)
        self._births: Dict[int, List[Trajectory]] = {}
    
    def add_traj(self, traj: Trajectory):
        """
        Adds an existing trajectory to the map and indexes it by birth time

        Parameters:
        - traj: Trajectory to add

        Raises:
        - ValueError: If the trajectory's start time is outside [0, self.size)
        """
        if not (0 <= traj.start < self.size):
            raise ValueError("Added trajectory is out of map's time-range")

        self.trajs.append(traj)
        self._births.setdefault(traj.start, []).append(traj)

    def generate(self, model: Model, max_active: int = 1000000, lambda_births: float = 0.05, PS: float = 0.99,
                 die_prob: float = 1., x0: Union[str, ArrayLike] = 'outside', cov: Optional[ArrayLike] = None,
                 min_len: int = 1, time_steps: Optional[int] = None):
        """
        Stochastically generates trajectories across the map horizon

        Parameters:
        - model: `Model` defining dynamics and noise
        - max_active: Maximum allowed simultaneous newly created trajectories
        - lambda_births: Poisson rate (expected births per time step)
        - PS: Probability of survival
        - die_prob: Probability a trajectory dies if it is outside the FOV
        - mean: Initial state behavior:
            - ArrayLike: mean of the initial state
            - 'outside': Random position outside radar FOV
            - 'inside': Random position inside radar FOV
        - cov: Covariance of the inital state
        - min_len: Minimum lenght of the trajectory
        - time_steps: Total number of time steps to generate, None if all
        """
        cur = 0
        cur_deaths = {}

        OUT_THR = 1.04

        def is_out(x: np.ndarray) -> bool:
            pos = model.get_pos(x)
            return np.linalg.norm(pos[:2] - self.center) > self.rad * OUT_THR

        def do_cont(x: np.ndarray) -> bool:
            if x is None or x.size == 0:
                return False
            rnd = self.rng.random()
            if is_out(x):
                return rnd > die_prob
            return True

        if time_steps is None:
            time_steps = self.size
        time_steps = min(int(time_steps), self.size)

        for t in range(time_steps):
            cur -= cur_deaths.get(t, 0)
            to_born = self.rng.poisson(lambda_births)
            to_born = max(0, min(to_born, max_active - cur))
            cur += to_born
    
            for _ in range(to_born):

                if isinstance(x0, (np.ndarray, list, tuple)):
                    if cov is None:
                        x = np.asarray(x0, dtype=model.dtype)
                    else:
                        x = self.rng.multivariate_normal(x0, cov).astype(model.dtype)
                    traj = Trajectory(model, x, min_len, start_time=t, rng=self.rng)
                elif isinstance(x0, str) and x0 == 'inside':
                    if cov is None:
                        x = np.zeros(model.state_dim, dtype=model.dtype)
                    else:
                        x = self.rng.multivariate_normal(x0, cov).astype(model.dtype)
                    model.get_pos(x)[:] = sample_circle(radius=self.rad, center=self.center, size=1, rng=self.rng, dtype=model.dtype)[0]
                    traj = Trajectory(model, x, min_len, start_time=t, rng=self.rng)
                elif isinstance(x0, str) and x0 == 'outside':
                    traj = self._get_outside_traj(model, min_len, t, cov=cov)
                else:
                    raise ValueError(f"Unsupported x0 value. Expected np.ndarray, 'inside' or 'outside', got {x0}")
                traj = self._extend_traj(traj, do_cont, PS)
                cur_deaths[traj.end] = cur_deaths.get(traj.end, 0) + 1
                self.add_traj(traj)
    
    def _get_outside_traj(self, model: Model, min_len: int, time: int, cov: Optional[ArrayLike] = None) -> Trajectory:
        """
        Generates a trajectory that starts outside the FOV and naturally travels 
        towards the radar edge based purely on the model's process noise logic
        """
        init_len = int(self.rng.random() * min_len) + 5
        x0 = np.zeros(model.state_dim, dtype=model.dtype)
        if cov is not None:
            x0 = self.rng.multivariate_normal(x0, cov).astype(model.dtype)
        traj = Trajectory(model, x0, init_len, start_time=time, rng=self.rng)

        start = model.get_pos(traj.state_at(time))[:2]
        end = model.get_pos(traj.state_at(-1))[:2]

        vec = start - end
        norm = max(1e-8, np.linalg.norm(vec))

        # projecting endpoint on to the FOV such that  start point is outside of the FOV
        point = vec / norm * self.rad

        # rotating by +- 1/5 pi
        angle = self.rng.uniform(-np.pi / 5., np.pi / 5.)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        point = R @ point

        # shifting coords so that end point is on the edge of FOV, heading invards
        shift = (point + self.center) - end
        traj.shift_coords(shift)
        
        return traj

    def _extend_traj(self, traj: Trajectory, do_cont: Callable, PS: float) -> Trajectory:
        """Extends the trajectory untils it dies as per do_cont"""
        ndat = self.rng.geometric(1 - PS) if PS < 1.0 else float('inf')
        ndat = min(self.size - traj.start, ndat)
        while traj.size < ndat and do_cont(traj.state_at(-1)):
            nlen = min(ndat - traj.size, 10)
            traj.extend(nlen)
        traj.shrink_to_fit()
        return traj

    class TrajMapIterator:
        """
        Time-stepping iterator yielding active trajectories in a TrajMap
        """

        def __init__(self, trajmap: "TrajMap"):
            """
            Create an iterator for `trajmap`

            Parameters:
            - trajmap: TrajMap instance to iterate over
            """
            self.map = trajmap
            self.time = -1
            self.active: List[Trajectory] = []

        def step(self):
            """Advances internal time and maintains the active trajectory min-heap"""
            self.time += 1

            while len(self.active) > 0 and self.active[0].end <= self.time:
                heappop(self.active)

            for traj in self.map._births.get(self.time, []):
                heappush(self.active, traj)

        def get_active(self) -> List[Trajectory]:
            """Returns the current active trajectories"""
            return self.active
        
        def at_end(self) -> bool:
            """Evaluates if the iterator has advanced past the map horizon"""
            return self.time >= self.map.size
        
        def __iter__(self) -> Iterator[Trajectory]:
            return self
        
        def __next__(self) -> List[Trajectory]:
            """Steps forward and returns the list of active trajectories"""
            self.step()
            if self.at_end():
                raise StopIteration
            return self.active

    def iterator(self) -> "TrajMap.TrajMapIterator":
        """Returns a fresh `TrajMapIterator` instance"""
        return TrajMap.TrajMapIterator(self)
    
    def __iter__(self) -> "TrajMap.TrajMapIterator":
        return self.iterator()
