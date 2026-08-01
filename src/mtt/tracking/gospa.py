"""
References:
- RAHMATHULLAH, Abu Sajana; GARCÍA-FERNÁNDEZ, Ángel F.; SVENSSON, Lennart. Generalized Optimal Sub-Pattern Assignment Metric. In: 2017 20th International Conference on Information Fusion (FUSION). Xi’an, China: IEEE, 2017, pp. 1–8. Available from doi: 10.23919/ICIF.2017.8009645

Inspired by:
- https://github.com/ewilthil/gospapy/blob/master/gospapy/gospa.py
"""

import numpy as np
from murty import assignment, assignment_sparse, AssignmentWorkers, ASSIGNMENT_EMPTY
from typing import Callable, Optional


def gospa(states: np.ndarray, estimates: np.ndarray, c: float, p: float, cost_function: Optional[Callable] = None,
          ass_workers: Optional[AssignmentWorkers] = None, sparse: bool = False):
    """
    Computes the gospa metric for given set of states and estimates

    Parameter:
    - states: Set of real states
    - estimates: Set of estimates of the states
    - c: Cut off distance
    - p: The order parameter
    - cost_function: Cost metric among state and estimate
    - ass_workers: Workers for the assignment solver
    - sparse: Whether to use sparse solver or not

    Returns:
    - Tuple(gospa, gospa_loc, gospa_miss, gospa_false)
    """

    alpha = 2.
    if c <= 0:
        raise ValueError("c must be from (0, inf)")
    if p < 1:
        raise ValueError("p must be from [1, inf)")

    nstates = len(states)
    nests = len(estimates)
    card_term = c**p / alpha
    max_cost = c**p

    if nstates == 0:
        gospa_false = card_term * nests
        return gospa_false, 0., 0., gospa_false
    elif nests == 0:
        gospa_missed = card_term * nstates
        return gospa_missed, 0., gospa_missed, 0.
    
    if cost_function is None:
        diff = states[:, np.newaxis, :] - estimates[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)**p
        C = np.minimum(dist, max_cost)
    else:
        C = np.zeros((nstates, nests))
        for i in range(nstates):
            for j in range(nests):
                C[i, j] = min(cost_function(states[i], estimates[j])**p, max_cost)
    C -= max_cost

    if sparse:
        ass = assignment_sparse(C, max_cost=-1e-12, workers=ass_workers)
    else:
        ass = assignment(C, workers=ass_workers)

    gospa_loc = 0.
    nass = 0

    for i, j in ass.ass:
        if i != ASSIGNMENT_EMPTY and j != ASSIGNMENT_EMPTY and C[i, j] < 0:
            gospa_loc += C[i, j] + max_cost
            nass += 1

    gospa_missed = card_term * (nstates - nass)
    gospa_false = card_term * (nests - nass)
    gospa = (gospa_loc + gospa_missed + gospa_false)
    return gospa, gospa_loc, gospa_missed, gospa_false
