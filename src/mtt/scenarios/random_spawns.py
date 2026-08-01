import numpy as np
from typing import Optional
import mtt
from ..utils.utils import get_rng


def scenario_random_spawn(ndat: int, model: mtt.Model, lambda_births: float, lambda_init_births: int = 0, radius: float = 150, PS: float = 0.995,
                          init_vel_var: float = 1., rng: Optional[np.random.Generator] = None, seed: int = None):
    """
    Creates Scenario with random spawns over the FOV

    Parameters:
    - ndat: Total number of time steps
    - model: Motion model
    - lambda_births: Expected number of newborn targets at each time step
    - lambda_init_births: Expected number of already existing targets
    - radius: Radius of the FOV
    - PS: Probability of survival at each time step
    - init_vel_var: Variance of the initial velocity
    - rng: Optional `np.random.Generator` (preferred over seed)
    - seed: Integer seed for RNG creation
    """

    rng = get_rng(rng=rng, seed=seed)

    b_mean = np.zeros((4,), dtype=np.float64)
    pos_var = (radius/5)**2
    b_cov = np.diag([pos_var, pos_var, init_vel_var, init_vel_var]).astype(dtype=np.float64)

    tmap = mtt.TrajMap(ndat, radius, rng=rng)

    # Initial targets
    tmap.generate(model, lambda_births=lambda_init_births, PS=PS, x0=b_mean, cov=b_cov, min_len=1, time_steps=1)
    
    # New born targets
    tmap.generate(model, lambda_births=lambda_births, PS=PS, x0=b_mean, cov=b_cov, min_len=10)

    return tmap
