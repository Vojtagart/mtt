from .. import _core
import numpy as np
from numpy.typing import DTypeLike
from .utils import _get_binding_class
from .mixtures import GaussianMixture


def PhdTracker(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, trunc_thr: float = 1e-5,
               merge_thr: float = 4.0, max_components: int = 250, conf_thr: float = 0.5,
               birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64):
    """
    Initializes a PHD Tracker

    Parameters:
    - birth_mixture: Birth mixture for prediction
    - F: Transition matrix (SDIM, SDIM)
    - Q: Process noise matrix (SDIM, SDIM)
    - H: Measurement matrix (MDIM, SDIM)
    - R: Measurement noise matrix (MDIM, MDIM)
    - PD: Probability of detection
    - PS: Probability of survival
    - clutter_int: Clutter intensity (lambda from PPP)
    - PG: Probability of being inside the gate
    - trunc_thr: Threshold below which components are dropped
    - merge_thr: Maximum squared Mahalanobis distance for merging
    - max_components: Maximum allowed components after merging
    - conf_thr: Weight threshold to consider a track confirmed
    - birth_mixture0: Birth mixture utilized for the first time step
    - dtype: Data type
    """

    sdim = F.shape[0]
    mdim = H.shape[0]

    kwargs = {
        "birth_mixture": birth_mixture,
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "PD": PD,
        "PS": PS,
        "clutter_int": clutter_int,
        "PG": PG,
        "trunc_thr": trunc_thr,
        "merge_thr": merge_thr,
        "max_components": max_components,
        "conf_thr": conf_thr,
        "birth_mixture0": birth_mixture0
    }

    cls = _get_binding_class("_PhdTracker", dtype, (sdim, mdim))
    return cls(**kwargs)


def PmbmTracker(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, min_poisson_weight: float = 4.0, max_hypothesis: int = 250,
               conf_thr: float = 0.5, recyclate: bool = True, merge_comps: bool = False, merge_thr: float = 0.0,
               sparsify: bool = True, max_per_row: int = 1000, birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64):
    """
    Initializes a PMBM Tracker

    Parameters:
    - birth_mixture: Birth mixture for prediction
    - F: Transition matrix (SDIM, SDIM)
    - Q: Process noise matrix (SDIM, SDIM)
    - H: Measurement matrix (MDIM, SDIM)
    - R: Measurement noise matrix (MDIM, MDIM)
    - PD: Probability of detection
    - PS: Probability of survival
    - clutter_int: Clutter intensity (lambda from PPP)
    - PG: Probability of being inside the gate
    - min_global_hypot_weight: Thresholf for global hypothesis to be considered
    - min_bernoulli_exist_prob: Threshold for Bernoulli existence probability
    - min_poisson_weight: Threshold for Poisson component weights
    - max_hypothesis: Maximum number of global hypothesis
    - conf_thr: Existence probability threshold to consider a track confirmed
    - recyclate: Whether to recyclate Bernoulli components instead of discarding them
    - merge_comps: Whether to merge similar components
    - merge_thr: Maximal distance (via KLD) to merge two Bernoulli components
    - sparsify: Whether to sparsify cost matrix
    - max_per_row: Maximum of elements per row in sparsified cost matrix
    - birth_mixture0: Birth mixture utilized for the first time step
    - dtype: Data type
    """

    sdim = F.shape[0]
    mdim = H.shape[0]

    kwargs = {
        "birth_mixture": birth_mixture,
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "PD": PD,
        "PS": PS,
        "clutter_int": clutter_int,
        "PG": PG,
        "min_global_hypot_weight": min_global_hypot_weight,
        "min_bernoulli_exist_prob": min_bernoulli_exist_prob,
        "min_poisson_weight": min_poisson_weight,
        "max_hypothesis": max_hypothesis,
        "conf_thr": conf_thr,
        "recyclate": recyclate,
        "merge_comps": merge_comps,
        "merge_thr": merge_thr,
        "sparsify": sparsify,
        "max_per_row": max_per_row,
        "birth_mixture0": birth_mixture0
    }

    cls = _get_binding_class("_PmbmTracker", dtype, (sdim, mdim))
    return cls(**kwargs)

def MbmTracker(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, max_hypothesis: int = 250,
               conf_thr: float = 0.5, merge_comps: bool = False, merge_thr: float = 0.0,
               sparsify: bool = True, max_per_row: int = 1000, birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64):
    """
    Initializes a MBM Tracker

    Parameters:
    - birth_mixture: Birth mixture for prediction
    - F: Transition matrix (SDIM, SDIM)
    - Q: Process noise matrix (SDIM, SDIM)
    - H: Measurement matrix (MDIM, SDIM)
    - R: Measurement noise matrix (MDIM, MDIM)
    - PD: Probability of detection
    - PS: Probability of survival
    - clutter_int: Clutter intensity (lambda from PPP)
    - PG: Probability of being inside the gate
    - min_global_hypot_weight: Thresholf for global hypothesis to be considered
    - min_bernoulli_exist_prob: Threshold for Bernoulli existence probability
    - max_hypothesis: Maximum number of global hypothesis
    - conf_thr: Existence probability threshold to consider a track confirmed
    - merge_comps: Whether to merge similar components
    - merge_thr: Maximal distance (via KLD) to merge two Bernoulli components
    - sparsify: Whether to sparsify cost matrix
    - max_per_row: Maximum of elements per row in sparsified cost matrix
    - birth_mixture0: Birth mixture utilized for the first time step
    - dtype: Data type
    """

    sdim = F.shape[0]
    mdim = H.shape[0]

    kwargs = {
        "birth_mixture": birth_mixture,
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "PD": PD,
        "PS": PS,
        "clutter_int": clutter_int,
        "PG": PG,
        "min_global_hypot_weight": min_global_hypot_weight,
        "min_bernoulli_exist_prob": min_bernoulli_exist_prob,
        "max_hypothesis": max_hypothesis,
        "conf_thr": conf_thr,
        "merge_comps": merge_comps,
        "merge_thr": merge_thr,
        "sparsify": sparsify,
        "max_per_row": max_per_row,
        "birth_mixture0": birth_mixture0
    }

    cls = _get_binding_class("_MbmTracker", dtype, (sdim, mdim))
    return cls(**kwargs)

def TombTracker(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, min_poisson_weight: float = 4.0, max_hypothesis: int = 250,
               conf_thr: float = 0.5, recyclate: bool = True, sparsify: bool = True, max_per_row: int = 1000,
               birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64):
    """
    Initializes a TOMB/P Tracker

    Parameters:
    - birth_mixture: Birth mixture for prediction
    - F: Transition matrix (SDIM, SDIM)
    - Q: Process noise matrix (SDIM, SDIM)
    - H: Measurement matrix (MDIM, SDIM)
    - R: Measurement noise matrix (MDIM, MDIM)
    - PD: Probability of detection
    - PS: Probability of survival
    - clutter_int: Clutter intensity (lambda from PPP)
    - PG: Probability of being inside the gate
    - min_global_hypot_weight: Thresholf for global hypothesis to be considered
    - min_bernoulli_exist_prob: Threshold for Bernoulli existence probability
    - min_poisson_weight: Threshold for Poisson component weights
    - max_hypothesis: Maximum number of global hypothesis
    - conf_thr: Existence probability threshold to consider a track confirmed
    - recyclate: Whether to recyclate Bernoulli components instead of discarding them
    - sparsify: Whether to sparsify cost matrix
    - max_per_row: Maximum of elements per row in sparsified cost matrix
    - birth_mixture0: Birth mixture utilized for the first time step
    - dtype: Data type
    """

    sdim = F.shape[0]
    mdim = H.shape[0]

    kwargs = {
        "birth_mixture": birth_mixture,
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "PD": PD,
        "PS": PS,
        "clutter_int": clutter_int,
        "PG": PG,
        "track_oriented": True,
        "min_global_hypot_weight": min_global_hypot_weight,
        "min_bernoulli_exist_prob": min_bernoulli_exist_prob,
        "min_poisson_weight": min_poisson_weight,
        "max_hypothesis": max_hypothesis,
        "conf_thr": conf_thr,
        "recyclate": recyclate,
        "sparsify": sparsify,
        "max_per_row": max_per_row,
        "birth_mixture0": birth_mixture0
    }

    cls = _get_binding_class("_PmbTracker", dtype, (sdim, mdim))
    return cls(**kwargs)

def MombTracker(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, alpha: float = 0.0, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, min_poisson_weight: float = 4.0, max_hypothesis: int = 250,
               conf_thr: float = 0.5, recyclate: bool = True, sparsify: bool = True, max_per_row: int = 1000,
               birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64):
    """
    Initializes a MOMB/P Tracker

    Parameters:
    - birth_mixture: Birth mixture for prediction
    - F: Transition matrix (SDIM, SDIM)
    - Q: Process noise matrix (SDIM, SDIM)
    - H: Measurement matrix (MDIM, SDIM)
    - R: Measurement noise matrix (MDIM, MDIM)
    - PD: Probability of detection
    - PS: Probability of survival
    - clutter_int: Clutter intensity (lambda from PPP)
    - PG: Probability of being inside the gate
    - alpha: Alpha parameter for the IMOMB/P
    - min_global_hypot_weight: Thresholf for global hypothesis to be considered
    - min_bernoulli_exist_prob: Threshold for Bernoulli existence probability
    - min_poisson_weight: Threshold for Poisson component weights
    - max_hypothesis: Maximum number of global hypothesis
    - conf_thr: Existence probability threshold to consider a track confirmed
    - recyclate: Whether to recyclate Bernoulli components instead of discarding them
    - sparsify: Whether to sparsify cost matrix
    - max_per_row: Maximum of elements per row in sparsified cost matrix
    - birth_mixture0: Birth mixture utilized for the first time step
    - dtype: Data type
    """

    sdim = F.shape[0]
    mdim = H.shape[0]

    kwargs = {
        "birth_mixture": birth_mixture,
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "PD": PD,
        "PS": PS,
        "clutter_int": clutter_int,
        "PG": PG,
        "alpha": alpha,
        "track_oriented": False,
        "min_global_hypot_weight": min_global_hypot_weight,
        "min_bernoulli_exist_prob": min_bernoulli_exist_prob,
        "min_poisson_weight": min_poisson_weight,
        "max_hypothesis": max_hypothesis,
        "conf_thr": conf_thr,
        "recyclate": recyclate,
        "sparsify": sparsify,
        "max_per_row": max_per_row,
        "birth_mixture0": birth_mixture0
    }

    cls = _get_binding_class("_PmbTracker", dtype, (sdim, mdim))
    return cls(**kwargs)
