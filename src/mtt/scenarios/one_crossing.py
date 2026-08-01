import numpy as np
from typing import Optional
import mtt
from ..utils.utils import get_rng


def scenario_one_crossing(ndat: int, ntargets: int, radius: float = 100, PS: float = 0.995, q: float = 0.01,
                          r: float = 1., init_vel_var: float = 1., rng: Optional[np.random.Generator] = None, seed: int = None):
    """
    Creates Scenario with multiple targets in close proximity at the middle time step

    Parameters:
    - ndat: Total number of time steps
    - ntargets: Initial number of targets, contained in the crossing
    - radius: Radius of the FOV
    - PS: Probability of survival at each time step
    - q: Process noise variance
    - r: Measurement noise variance
    - init_vel_var: Variance for the initial velocity
    - rng: Optional `np.random.Generator` (preferred over seed)
    - seed: Integer seed for RNG creation
    """

    if ndat < 10:
        raise ValueError("ndat must be atleast 10")

    rng = get_rng(rng=rng, seed=seed)
    model = mtt.Model.CVM(dt=1, q=q, r=r)

    # Targets located at (0,0), shifted afterwards to make crossing
    b_mean = np.zeros((4,), dtype=np.float64)
    b_cov = np.diag([0, 0, init_vel_var, init_vel_var]).astype(dtype=np.float64)

    # Used to sample the crossing position - around (0,0)
    c_mean = np.zeros((2,), dtype=np.float64)
    c_cov = 10*q * np.eye(2, dtype=np.float64)
    cross_t = ndat // 2 - 1

    x0s = rng.multivariate_normal(b_mean, b_cov, size=ntargets)
    cs = rng.multivariate_normal(c_mean, c_cov, size=ntargets)
    if PS >= 1.0:
        times = np.full(ntargets, ndat)
    else:
        times = rng.geometric(1 - PS, size=ntargets)

    tmap = mtt.TrajMap(ndat, radius, rng=rng)

    # Generate initial crossing targets
    for i in range(ntargets):
        traj = mtt.Trajectory(model, x0s[i], ndat=ndat, rng=rng)
        state = traj.state_at(cross_t)
        to_shift = cs[i] - model.get_pos(state)
        traj.shift_coords(to_shift)
        total_t = min(ndat, times[i])
        traj.resize(total_t)
        tmap.add_traj(traj)

    return tmap
