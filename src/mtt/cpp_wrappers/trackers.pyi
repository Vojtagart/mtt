from typing import Tuple
import numpy as np
from numpy.typing import DTypeLike

from mtt.cpp_wrappers.mixtures import GaussianMixture, MultiBernoulli


class PhdTracker:
    """
    PHD tracker
    """

    def __init__(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, trunc_thr: float = 1e-5,
               merge_thr: float = 4.0, max_components: int = 250, conf_thr: float = 0.5,
               birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64) -> None:
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
        ...

    def step(self, measurements: np.ndarray) -> None:
        """
        Performs a prediction, update and prunning
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prediction(self) -> None:
        """
        Performs a prediction
        """
        ...

    def update(self, measurements: np.ndarray) -> None:
        """
        Performs an update
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def confirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts currently confirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (only 1 supported)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM))
        """
        ...

    def unconfirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts currently unconfirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (only 1 supported)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM))
        """
        ...

    def reset(self) -> None:
        """
        Resets the internal state of tracker
        """
        ...

    @property
    def mixture(self) -> GaussianMixture:
        """Current posterior mixture"""
        ...

    @mixture.setter
    def mixture(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture(self) -> GaussianMixture:
        """Current birth mixture"""
        ...
    @birth_mixture.setter
    def birth_mixture(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture0(self) -> GaussianMixture:
        """Current step 0 birth mixture"""
        ...
    @birth_mixture0.setter
    def birth_mixture0(self, value: GaussianMixture) -> None:
        ...

    @property
    def PD(self) -> float:
        """Probability of detection"""
        ...
    @PD.setter
    def PD(self, value: float) -> None: ...

    @property
    def PS(self) -> float: 
        """Probability of survival"""
        ...
    @PS.setter
    def PS(self, value: float) -> None: ...

    @property
    def clutter_int(self) -> float:
        """Clutter intensity"""
        ...
    @clutter_int.setter
    def clutter_int(self, value: float) -> None:
        ...

    @property
    def PG(self) -> float:
        """Probability of being inside the gate"""
        ...
    @PG.setter
    def PG(self, value: float) -> None: ...

    @property
    def trunc_thr(self) -> float:
        """Truncation threshold"""
        ...
    @trunc_thr.setter
    def trunc_thr(self, value: float) -> None: ...

    @property
    def merge_thr(self) -> float:
        """Merge threshold"""
        ...
    @merge_thr.setter
    def merge_thr(self, value: float) -> None: ...

    @property
    def max_components(self) -> int:
        """Maximum number of components"""
        ...
    @max_components.setter
    def max_components(self, value: int) -> None: ...

    @property
    def conf_thr(self) -> float:
        """Confirmation threshold"""
        ...
    @conf_thr.setter
    def conf_thr(self, value: float) -> None: ...

    @property
    def t0(self) -> bool:
        """Whether is tracker in time step 0"""
        ...
    @t0.setter
    def t0(self, value: bool) -> None: ...


class PmbmTracker:
    """
    PMBM tracker
    """

    def __init__(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, min_poisson_weight: float = 4.0, max_hypothesis: int = 250,
               conf_thr: float = 0.5, recyclate: bool = True, merge_comps: bool = False, merge_thr: float = 0.0,
               sparsify: bool = True, max_per_row: int = 1000, birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64) -> None:
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
        ...

    def step(self, measurements: np.ndarray) -> None:
        """
        Performs a prediction, update and prunning
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prediction(self) -> None:
        """
        Performs a prediction
        """
        ...

    def update(self, measurements: np.ndarray) -> None:
        """
        Performs an update
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prune(self) -> None:
        """
        Performs pruning
        """
        ...

    def confirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently confirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (1 or 2)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...

    def unconfirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently unconfirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (1 or 2)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...
    
    def reset(self) -> None:
        """
        Resets the internal state of tracker
        """
        ...

    @property
    def poiss(self) -> GaussianMixture:
        """Current Poisson part"""
        ...

    @poiss.setter
    def poiss(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture(self) -> GaussianMixture:
        """Current birth mixture"""
        ...
    @birth_mixture.setter
    def birth_mixture(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture0(self) -> GaussianMixture:
        """Current step 0 birth mixture"""
        ...
    @birth_mixture0.setter
    def birth_mixture0(self, value: GaussianMixture) -> None:
        ...

    @property
    def PD(self) -> float:
        """Probability of detection"""
        ...
    @PD.setter
    def PD(self, value: float) -> None: ...

    @property
    def PS(self) -> float: 
        """Probability of survival"""
        ...
    @PS.setter
    def PS(self, value: float) -> None: ...

    @property
    def clutter_int(self) -> float:
        """Clutter intensity"""
        ...
    @clutter_int.setter
    def clutter_int(self, value: float) -> None:
        ...

    @property
    def PG(self) -> float:
        """Probability of being inside the gate"""
        ...
    @PG.setter
    def PG(self, value: float) -> None: ...

    @property
    def min_global_hypot_weight(self) -> float:
        """Minimum Global hypothesis weight"""
        ...
    @min_global_hypot_weight.setter
    def min_global_hypot_weight(self, value: float) -> None: ...

    @property
    def min_bernoulli_exist_prob(self) -> float:
        """Minimum Bernoulli existence probability"""
        ...
    @min_bernoulli_exist_prob.setter
    def min_bernoulli_exist_prob(self, value: float) -> None: ...

    @property
    def min_poisson_weight(self) -> float:
        """Minimal Poisson weight"""
        ...
    @min_poisson_weight.setter
    def min_poisson_weight(self, value: float) -> None: ...

    @property
    def max_hypothesis(self) -> int:
        """Maximum number of global hypothesis"""
        ...
    @max_hypothesis.setter
    def max_hypothesis(self, value: int) -> None: ...

    @property
    def conf_thr(self) -> float:
        """Confirmation threshold"""
        ...
    @conf_thr.setter
    def conf_thr(self, value: float) -> None: ...

    @property
    def recyclate(self) -> bool:
        """Whether recyclate Bernoulli components"""
        ...
    @recyclate.setter
    def recyclate(self, value: bool) -> None: ...

    @property
    def merge_comps(self) -> bool:
        """Whether similar components are merged"""
        ...
    @merge_comps.setter
    def merge_comps(self, value: bool) -> None: ...

    @property
    def merge_thr(self) -> float:
        """Minimum distance to merge 2 Bernoulli components"""
        ...
    @merge_thr.setter
    def confmerge_thr_thr(self, value: float) -> None: ...

    @property
    def t0(self) -> bool:
        """Whether is tracker in time step 0"""
        ...
    @t0.setter
    def t0(self, value: bool) -> None: ...

    @property
    def sparse(self) -> bool:
        """Whether sparse cost matrix"""
        ...
    @sparse.setter
    def sparse(self, value: bool) -> None: ...

    @property
    def max_per_row(self) -> int:
        """Maximum elements per row in sparsified cost matrix"""
        ...
    @max_per_row.setter
    def max_per_row(self, value: int) -> None: ...


class MbmTracker:
    """
    MBM tracker
    """

    def __init__(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, max_hypothesis: int = 250,
               conf_thr: float = 0.5, merge_comps: bool = False, merge_thr: float = 0.0,
               sparsify: bool = True, max_per_row: int = 1000, birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64) -> None:
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
        ...

    def step(self, measurements: np.ndarray) -> None:
        """
        Performs a prediction, update and prunning
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prediction(self) -> None:
        """
        Performs a prediction
        """
        ...

    def update(self, measurements: np.ndarray) -> None:
        """
        Performs an update
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prune(self) -> None:
        """
        Performs pruning
        """
        ...

    def confirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently confirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (1 or 2)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...

    def unconfirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently unconfirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (1 or 2)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...
    
    def reset(self) -> None:
        """
        Resets the internal state of tracker
        """
        ...

    @property
    def birth_mixture(self) -> MultiBernoulli:
        """Current birth mixture"""
        ...
    @birth_mixture.setter
    def birth_mixture(self, value: MultiBernoulli) -> None:
        ...

    @property
    def birth_mixture0(self) -> GaussianMixture:
        """Current step 0 birth mixture"""
        ...
    @birth_mixture0.setter
    def birth_mixture0(self, value: GaussianMixture) -> None:
        ...

    @property
    def PD(self) -> float:
        """Probability of detection"""
        ...
    @PD.setter
    def PD(self, value: float) -> None: ...

    @property
    def PS(self) -> float: 
        """Probability of survival"""
        ...
    @PS.setter
    def PS(self, value: float) -> None: ...

    @property
    def clutter_int(self) -> float:
        """Clutter intensity"""
        ...
    @clutter_int.setter
    def clutter_int(self, value: float) -> None:
        ...

    @property
    def PG(self) -> float:
        """Probability of being inside the gate"""
        ...
    @PG.setter
    def PG(self, value: float) -> None: ...

    @property
    def min_global_hypot_weight(self) -> float:
        """Minimum Global hypothesis weight"""
        ...
    @min_global_hypot_weight.setter
    def min_global_hypot_weight(self, value: float) -> None: ...

    @property
    def min_bernoulli_exist_prob(self) -> float:
        """Minimum Bernoulli existence probability"""
        ...
    @min_bernoulli_exist_prob.setter
    def min_bernoulli_exist_prob(self, value: float) -> None: ...

    @property
    def max_hypothesis(self) -> int:
        """Maximum number of global hypothesis"""
        ...
    @max_hypothesis.setter
    def max_hypothesis(self, value: int) -> None: ...

    @property
    def conf_thr(self) -> float:
        """Confirmation threshold"""
        ...
    @conf_thr.setter
    def conf_thr(self, value: float) -> None: ...

    @property
    def merge_comps(self) -> bool:
        """Whether similar components are merged"""
        ...
    @merge_comps.setter
    def merge_comps(self, value: bool) -> None: ...

    @property
    def merge_thr(self) -> float:
        """Minimum distance to merge 2 Bernoulli components"""
        ...
    @merge_thr.setter
    def confmerge_thr_thr(self, value: float) -> None: ...

    @property
    def t0(self) -> bool:
        """Whether is tracker in time step 0"""
        ...
    @t0.setter
    def t0(self, value: bool) -> None: ...

    @property
    def sparse(self) -> bool:
        """Whether sparse cost matrix"""
        ...
    @sparse.setter
    def sparse(self, value: bool) -> None: ...

    @property
    def max_per_row(self) -> int:
        """Maximum elements per row in sparsified cost matrix"""
        ...
    @max_per_row.setter
    def max_per_row(self, value: int) -> None: ...

class TombTracker:
    """
    TOMB/P tracker
    """

    def __init__(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, min_poisson_weight: float = 4.0, max_hypothesis: int = 250,
               conf_thr: float = 0.5, recyclate: bool = True, sparsify: bool = True, max_per_row: int = 1000,
               birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64) -> None:
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
        ...

    def step(self, measurements: np.ndarray) -> None:
        """
        Performs a prediction, update and prunning
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prediction(self) -> None:
        """
        Performs a prediction
        """
        ...

    def update(self, measurements: np.ndarray) -> None:
        """
        Performs an update
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prune(self) -> None:
        """
        Performs pruning
        """
        ...

    def confirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently confirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (only 1 supported)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...

    def unconfirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently unconfirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (only 1 supported)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...
    
    def reset(self) -> None:
        """
        Resets the internal state of tracker
        """
        ...

    @property
    def poiss(self) -> GaussianMixture:
        """Current Poisson part"""
        ...

    @poiss.setter
    def poiss(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture(self) -> GaussianMixture:
        """Current birth mixture"""
        ...
    @birth_mixture.setter
    def birth_mixture(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture0(self) -> GaussianMixture:
        """Current step 0 birth mixture"""
        ...
    @birth_mixture0.setter
    def birth_mixture0(self, value: GaussianMixture) -> None:
        ...

    @property
    def PD(self) -> float:
        """Probability of detection"""
        ...
    @PD.setter
    def PD(self, value: float) -> None: ...

    @property
    def PS(self) -> float: 
        """Probability of survival"""
        ...
    @PS.setter
    def PS(self, value: float) -> None: ...

    @property
    def clutter_int(self) -> float:
        """Clutter intensity"""
        ...
    @clutter_int.setter
    def clutter_int(self, value: float) -> None:
        ...

    @property
    def PG(self) -> float:
        """Probability of being inside the gate"""
        ...
    @PG.setter
    def PG(self, value: float) -> None: ...

    @property
    def min_global_hypot_weight(self) -> float:
        """Minimum Global hypothesis weight"""
        ...
    @min_global_hypot_weight.setter
    def min_global_hypot_weight(self, value: float) -> None: ...

    @property
    def min_bernoulli_exist_prob(self) -> float:
        """Minimum Bernoulli existence probability"""
        ...
    @min_bernoulli_exist_prob.setter
    def min_bernoulli_exist_prob(self, value: float) -> None: ...

    @property
    def min_poisson_weight(self) -> float:
        """Minimal Poisson weight"""
        ...
    @min_poisson_weight.setter
    def min_poisson_weight(self, value: float) -> None: ...

    @property
    def max_hypothesis(self) -> int:
        """Maximum number of global hypothesis"""
        ...
    @max_hypothesis.setter
    def max_hypothesis(self, value: int) -> None: ...

    @property
    def conf_thr(self) -> float:
        """Confirmation threshold"""
        ...
    @conf_thr.setter
    def conf_thr(self, value: float) -> None: ...

    @property
    def recyclate(self) -> bool:
        """Whether recyclate Bernoulli components"""
        ...
    @recyclate.setter
    def recyclate(self, value: bool) -> None: ...

    @property
    def t0(self) -> bool:
        """Whether is tracker in time step 0"""
        ...
    @t0.setter
    def t0(self, value: bool) -> None: ...

    @property
    def sparse(self) -> bool:
        """Whether sparse cost matrix"""
        ...
    @sparse.setter
    def sparse(self, value: bool) -> None: ...

    @property
    def max_per_row(self) -> int:
        """Maximum elements per row in sparsified cost matrix"""
        ...
    @max_per_row.setter
    def max_per_row(self, value: int) -> None: ...

class MombTracker:
    """
    MOMB/P tracker
    """

    def __init__(birth_mixture: GaussianMixture, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, PD: float,
               PS: float, clutter_int: float, PG: float = 0.999, alpha: float = 0.0, min_global_hypot_weight: float = 1e-5,
               min_bernoulli_exist_prob: float = 1e-5, min_poisson_weight: float = 4.0, max_hypothesis: int = 250,
               conf_thr: float = 0.5, recyclate: bool = True, sparsify: bool = True, max_per_row: int = 1000,
               birth_mixture0: GaussianMixture = None, dtype: DTypeLike = np.float64) -> None:
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
        ...

    def step(self, measurements: np.ndarray) -> None:
        """
        Performs a prediction, update and prunning
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prediction(self) -> None:
        """
        Performs a prediction
        """
        ...

    def update(self, measurements: np.ndarray) -> None:
        """
        Performs an update
        
        Parameters:
        - measurements: 2D array of measurements (N, MDIM)
        """
        ...

    def prune(self) -> None:
        """
        Performs pruning
        """
        ...

    def confirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently confirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (only 1 supported)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...

    def unconfirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts currently unconfirmed tracks based on the confirmation_threshold

        Parameters:
        - estimator_type: Type of the estimator (only 1 supported)

        Return:
        - Tuple (means (N, SDIM), covariances (N, SDIM, SDIM), ids (N,), track ids (N,))
        """
        ...
    
    def reset(self) -> None:
        """
        Resets the internal state of tracker
        """
        ...

    @property
    def poiss(self) -> GaussianMixture:
        """Current Poisson part"""
        ...

    @poiss.setter
    def poiss(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture(self) -> GaussianMixture:
        """Current birth mixture"""
        ...
    @birth_mixture.setter
    def birth_mixture(self, value: GaussianMixture) -> None:
        ...

    @property
    def birth_mixture0(self) -> GaussianMixture:
        """Current step 0 birth mixture"""
        ...
    @birth_mixture0.setter
    def birth_mixture0(self, value: GaussianMixture) -> None:
        ...

    @property
    def PD(self) -> float:
        """Probability of detection"""
        ...
    @PD.setter
    def PD(self, value: float) -> None: ...

    @property
    def PS(self) -> float: 
        """Probability of survival"""
        ...
    @PS.setter
    def PS(self, value: float) -> None: ...

    @property
    def clutter_int(self) -> float:
        """Clutter intensity"""
        ...
    @clutter_int.setter
    def clutter_int(self, value: float) -> None:
        ...

    @property
    def PG(self) -> float:
        """Probability of being inside the gate"""
        ...
    @PG.setter
    def PG(self, value: float) -> None: ...

    @property
    def alpha(self) -> float:
        """Alpha for IMOMB/P"""
        ...
    @alpha.setter
    def alpha(self, value: float) -> None: ...

    @property
    def min_global_hypot_weight(self) -> float:
        """Minimum Global hypothesis weight"""
        ...
    @min_global_hypot_weight.setter
    def min_global_hypot_weight(self, value: float) -> None: ...

    @property
    def min_bernoulli_exist_prob(self) -> float:
        """Minimum Bernoulli existence probability"""
        ...
    @min_bernoulli_exist_prob.setter
    def min_bernoulli_exist_prob(self, value: float) -> None: ...

    @property
    def min_poisson_weight(self) -> float:
        """Minimal Poisson weight"""
        ...
    @min_poisson_weight.setter
    def min_poisson_weight(self, value: float) -> None: ...

    @property
    def max_hypothesis(self) -> int:
        """Maximum number of global hypothesis"""
        ...
    @max_hypothesis.setter
    def max_hypothesis(self, value: int) -> None: ...

    @property
    def conf_thr(self) -> float:
        """Confirmation threshold"""
        ...
    @conf_thr.setter
    def conf_thr(self, value: float) -> None: ...

    @property
    def recyclate(self) -> bool:
        """Whether recyclate Bernoulli components"""
        ...
    @recyclate.setter
    def recyclate(self, value: bool) -> None: ...

    @property
    def t0(self) -> bool:
        """Whether is tracker in time step 0"""
        ...
    @t0.setter
    def t0(self, value: bool) -> None: ...

    @property
    def sparse(self) -> bool:
        """Whether sparse cost matrix"""
        ...
    @sparse.setter
    def sparse(self, value: bool) -> None: ...

    @property
    def max_per_row(self) -> int:
        """Maximum elements per row in sparsified cost matrix"""
        ...
    @max_per_row.setter
    def max_per_row(self, value: int) -> None: ...
