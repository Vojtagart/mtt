import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Tracker(ABC):
    """
    Abstract base class defining the standard interface for multi-target trackers

    Ensures that any derived tracking algorithm implements the required lifecycle 
    methods (update) and state accessors (confirmed/unconfirmed tracks) 
    expected by the visualization and evaluation helpers
    """

    @abstractmethod
    def step(self, dets: np.ndarray):
        """
        Updates the tracker state with a new set of measurements

        Parameters:
        - dets: Detections array of shape (N, D), where N is the number of 
          detections and D is the measurement dimension
        """
        ...

    @abstractmethod
    def confirmed_tracks(self, estimator_type: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the state estimates of all confirmed, actively tracked targets

        Parameters:
        - estimator_type: Type of the estimator

        Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple `(means, covariances)`
          `means` is an array of shape (N, D) and `covariances` is an array 
          of shape (N, D, D), representing the state of N confirmed targets
        """
        ...
    
    def unconfirmed_tracks(self, estimator_type: int = 1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieves the state estimates of all unconfirmed (tentative) targets

        Parameters:
        - estimator_type: Type of the estimator

        Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple `(means, covariances)`
          `means` is an array of shape (N, D) and `covariances` is an array
          of shape (N, D, D), representing the state of N unconfirmed targets
        """
        return None
        