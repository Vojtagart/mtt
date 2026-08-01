import pytest
import numpy as np
from mtt import PhdTracker, GaussianMixture

SDIMS = [2, 3, 4]
MDIMS = [2]
DTYPES = [np.float64]

def get_tracker_inputs(sdim, mdim, dtype):

    birth_mixture = GaussianMixture(sdim, dtype=dtype)
    birth_mixture.push(0.1, np.zeros(sdim, dtype=dtype), np.eye(sdim, dtype=dtype))
    
    F = np.eye(sdim, dtype=dtype)
    Q = np.eye(sdim, dtype=dtype) * 0.1
    
    H = np.eye(mdim, sdim, dtype=dtype)
    R = np.eye(mdim, dtype=dtype) * 0.1
    
    return birth_mixture, F, Q, H, R

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_init(sdim, mdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    
    tracker = PhdTracker(
        birth_mixture=birth,
        F=F, Q=Q, H=H, R=R,
        PD=0.95, PS=0.99, clutter_int=1e-3,
        PG=0.99, trunc_thr=1e-5, merge_thr=4.0,
        max_components=100, conf_thr=0.6,
        birth_mixture0 = birth, dtype=dtype
    )
    assert tracker is not None
    assert tracker.mixture.size() == 0

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_properties(sdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, 2, dtype)
    tracker = PhdTracker(birth, F, Q, H, R, 0.9, 0.9, 1e-4, dtype=dtype)
    
    tracker.PD = 0.8
    assert np.isclose(tracker.PD, 0.8)
    
    tracker.clutter_int = 1e-2
    assert np.isclose(tracker.clutter_int, 1e-2)
    
    tracker.max_components = 50
    assert tracker.max_components == 50

    new_birth = GaussianMixture(sdim, dtype=dtype)
    new_birth.push(0.5, np.ones(sdim, dtype=dtype), np.eye(sdim, dtype=dtype))
    tracker.birth_mixture = new_birth
    
    assert tracker.birth_mixture.size() == 1
    assert np.allclose(tracker.birth_mixture.M[0], np.ones(sdim, dtype=dtype))
    assert np.allclose(tracker.birth_mixture.C[0], np.eye(sdim, dtype=dtype))

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_update_and_tracks(sdim, mdim, dtype):

    rng = np.random.default_rng(42)
    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    tracker = PhdTracker(birth, F, Q, H, R, 0.95, 0.99, 1e-4, dtype=dtype)
    
    for _ in range(20):
        
        n_meas = rng.integers(0, 40)
        if n_meas > 0:
            Z = rng.random((n_meas, mdim)).astype(dtype)
        else:
            Z = np.zeros((0, mdim), dtype=dtype)
        tracker.step(Z)

        assert tracker.mixture.size() >= 0
        assert tracker.mixture.size() <= tracker.max_components

        means, covs = tracker.confirmed_tracks()
    
        assert type(means) is np.ndarray
        assert type(covs) is np.ndarray
        
        assert means.shape[1] == sdim
        assert covs.shape[1:] == (sdim, sdim)
