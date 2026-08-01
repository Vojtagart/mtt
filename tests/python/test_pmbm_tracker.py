import pytest
import numpy as np
from mtt import PmbmTracker, MbmTracker, GaussianMixture, MultiBernoulli, TombTracker, MombTracker

SDIMS = [2, 3, 4, 6]
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

# --------------------------------  PMBM  ----------------------------------------

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_init(sdim, mdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    
    tracker = PmbmTracker(
        birth_mixture=birth,
        F=F, Q=Q, H=H, R=R,
        PD=0.95, PS=0.99, clutter_int=1e-3,
        PG=0.99, min_global_hypot_weight=1e-5, min_bernoulli_exist_prob=1e-5, min_poisson_weight=1e-5,
        max_hypothesis=200, conf_thr=0.5, recyclate=True, merge_comps=True, merge_thr=1.4,
        sparsify=True, max_per_row=25, birth_mixture0 = birth, dtype=dtype
    )
    assert tracker is not None
    assert tracker.poiss.size() == 0

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_properties(sdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, 2, dtype)
    tracker = PmbmTracker(birth, F, Q, H, R, 0.9, 0.99, 1e-4, dtype=dtype)
    
    tracker.PD = 0.8
    assert np.isclose(tracker.PD, 0.8)
    
    tracker.clutter_int = 1e-2
    assert np.isclose(tracker.clutter_int, 1e-2)
    
    tracker.max_hypothesis = 50
    assert tracker.max_hypothesis == 50

    tracker.recyclate = False
    assert not tracker.recyclate

    tracker.merge_comps = False
    assert not tracker.merge_comps

    tracker.merge_thr = 1.
    assert np.isclose(tracker.merge_thr, 1.)

    new_birth = GaussianMixture(sdim, dtype=dtype)
    new_birth.push(0.5, np.ones(sdim, dtype=dtype), np.eye(sdim, dtype=dtype))
    tracker.birth_mixture = new_birth
    
    assert tracker.birth_mixture.size() == 1
    assert np.allclose(tracker.birth_mixture.M[0], np.ones(sdim, dtype=dtype))
    assert np.allclose(tracker.birth_mixture.C[0], np.eye(sdim, dtype=dtype))

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("merge_comps", [True, False])
@pytest.mark.parametrize("merge_thr", [0., .1, 1., 5.])
@pytest.mark.parametrize("sparsify", [True, False])
@pytest.mark.parametrize("max_per_row", [25])
def test_tracker_update_and_tracks(sdim, mdim, dtype, merge_comps, merge_thr, sparsify, max_per_row):

    rng = np.random.default_rng(42)
    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    tracker = PmbmTracker(birth, F, Q, H, R, 0.95, 0.99, 1e-4,
                          merge_comps=merge_comps, merge_thr=merge_thr, dtype=dtype,
                          sparsify=sparsify, max_per_row=max_per_row)
    
    for _ in range(20):
        
        n_meas = rng.integers(0, 40)
        if n_meas > 0:
            Z = rng.random((n_meas, mdim)).astype(dtype)
        else:
            Z = np.zeros((0, mdim), dtype=dtype)
        tracker.step(Z)

        means, covs, ids = tracker.confirmed_tracks()
    
        assert type(means) is np.ndarray
        assert type(covs) is np.ndarray
        assert type(ids) is np.ndarray
        
        assert means.shape[1] == sdim
        assert covs.shape[1:] == (sdim, sdim)

# --------------------------------  MBM  ----------------------------------------

def gm_to_mb(gm):
    ret = MultiBernoulli(gm.M.shape[1])
    for i in range(gm.size()):
        ret.push(max(1, gm.W[i]), gm.M[i], gm.C[i])
    return ret

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mbm_tracker_init(sdim, mdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    birth = gm_to_mb(birth)
    
    tracker = MbmTracker(
        birth_mixture=birth,
        F=F, Q=Q, H=H, R=R,
        PD=0.95, PS=0.99, clutter_int=1e-3,
        PG=0.99, min_global_hypot_weight=1e-5, min_bernoulli_exist_prob=1e-5,
        max_hypothesis=200, conf_thr=0.6, merge_comps=True, merge_thr=0.0,
        sparsify=True, max_per_row=25, birth_mixture0 = birth, dtype=dtype
    )
    assert tracker is not None

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_properties(sdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, 2, dtype)
    birth = gm_to_mb(birth)
    tracker = MbmTracker(birth, F, Q, H, R, 0.9, 0.9, 1e-4, 1e-4, dtype=dtype)
    
    tracker.PD = 0.8
    assert np.isclose(tracker.PD, 0.8)
    
    tracker.clutter_int = 1e-2
    assert np.isclose(tracker.clutter_int, 1e-2)
    
    tracker.max_hypothesis = 50
    assert tracker.max_hypothesis == 50

    tracker.merge_comps = False
    assert not tracker.merge_comps

    tracker.merge_thr = 1.
    assert np.isclose(tracker.merge_thr, 1.)

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("merge_comps", [True, False])
@pytest.mark.parametrize("merge_thr", [0., .1, 1., 5.])
@pytest.mark.parametrize("sparsify", [True, False])
@pytest.mark.parametrize("max_per_row", [25])
def test_tracker_update_and_tracks(sdim, mdim, dtype, merge_comps, merge_thr, sparsify, max_per_row):

    rng = np.random.default_rng(42)
    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    birth = gm_to_mb(birth)
    tracker = MbmTracker(birth, F, Q, H, R, 0.95, 0.99, 1e-4,
                         merge_comps=merge_comps, merge_thr=merge_thr, dtype=dtype,
                         sparsify=sparsify, max_per_row=max_per_row)
    
    for _ in range(20):
        
        n_meas = rng.integers(0, 40)
        if n_meas > 0:
            Z = rng.random((n_meas, mdim)).astype(dtype)
        else:
            Z = np.zeros((0, mdim), dtype=dtype)
        tracker.step(Z)

        means, covs, ids = tracker.confirmed_tracks()
    
        assert type(means) is np.ndarray
        assert type(covs) is np.ndarray
        assert type(ids) is np.ndarray
        
        assert means.shape[1] == sdim
        assert covs.shape[1:] == (sdim, sdim)

# --------------------------------  TOMB  ----------------------------------------

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_init(sdim, mdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    
    tracker = TombTracker(
        birth_mixture=birth,
        F=F, Q=Q, H=H, R=R,
        PD=0.95, PS=0.99, clutter_int=1e-3,
        PG=0.99, min_global_hypot_weight=1e-5, min_bernoulli_exist_prob=1e-5, min_poisson_weight=1e-5,
        max_hypothesis=200, conf_thr=0.6, recyclate=True,
        sparsify=True, max_per_row=25, birth_mixture0 = birth, dtype=dtype
    )
    assert tracker is not None
    assert tracker.poiss.size() == 0

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_properties(sdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, 2, dtype)
    tracker = TombTracker(birth, F, Q, H, R, 0.9, 0.99, 1e-4, dtype=dtype)
    
    tracker.PD = 0.8
    assert np.isclose(tracker.PD, 0.8)
    
    tracker.clutter_int = 1e-2
    assert np.isclose(tracker.clutter_int, 1e-2)
    
    tracker.max_hypothesis = 50
    assert tracker.max_hypothesis == 50

    tracker.recyclate = False
    assert not tracker.recyclate

    new_birth = GaussianMixture(sdim, dtype=dtype)
    new_birth.push(0.5, np.ones(sdim, dtype=dtype), np.eye(sdim, dtype=dtype))
    tracker.birth_mixture = new_birth
    
    assert tracker.birth_mixture.size() == 1
    assert np.allclose(tracker.birth_mixture.M[0], np.ones(sdim, dtype=dtype))
    assert np.allclose(tracker.birth_mixture.C[0], np.eye(sdim, dtype=dtype))

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sparsify", [True, False])
@pytest.mark.parametrize("max_per_row", [25])
def test_tracker_update_and_tracks(sdim, mdim, dtype, sparsify, max_per_row):

    rng = np.random.default_rng(42)
    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    tracker = TombTracker(birth, F, Q, H, R, 0.95, 0.99, 1e-4, dtype=dtype,
                          sparsify=sparsify, max_per_row=max_per_row)
    
    for _ in range(20):
        
        n_meas = rng.integers(0, 40)
        if n_meas > 0:
            Z = rng.random((n_meas, mdim)).astype(dtype)
        else:
            Z = np.zeros((0, mdim), dtype=dtype)
        tracker.step(Z)

        means, covs, ids = tracker.confirmed_tracks()
    
        assert type(means) is np.ndarray
        assert type(covs) is np.ndarray
        assert type(ids) is np.ndarray
        
        assert means.shape[1] == sdim
        assert covs.shape[1:] == (sdim, sdim)

# --------------------------------  MOMB  ----------------------------------------

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_init(sdim, mdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    
    tracker = MombTracker(
        birth_mixture=birth,
        F=F, Q=Q, H=H, R=R,
        PD=0.95, PS=0.99, clutter_int=1e-3,
        PG=0.99, alpha=0.0, min_global_hypot_weight=1e-5, min_bernoulli_exist_prob=1e-5, min_poisson_weight=1e-5,
        max_hypothesis=200, conf_thr=0.6, recyclate=True,
        sparsify=True, max_per_row=25, birth_mixture0 = birth, dtype=dtype
    )
    assert tracker is not None
    assert tracker.poiss.size() == 0

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_properties(sdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, 2, dtype)
    tracker = MombTracker(birth, F, Q, H, R, 0.9, 0.99, 1e-4, dtype=dtype)
    
    tracker.PD = 0.8
    assert np.isclose(tracker.PD, 0.8)
    
    tracker.clutter_int = 1e-2
    assert np.isclose(tracker.clutter_int, 1e-2)
    
    tracker.max_hypothesis = 50
    assert tracker.max_hypothesis == 50

    tracker.recyclate = False
    assert not tracker.recyclate

    tracker.alpha = 0.5
    assert np.isclose(tracker.alpha, 0.5)

    new_birth = GaussianMixture(sdim, dtype=dtype)
    new_birth.push(0.5, np.ones(sdim, dtype=dtype), np.eye(sdim, dtype=dtype))
    tracker.birth_mixture = new_birth
    
    assert tracker.birth_mixture.size() == 1
    assert np.allclose(tracker.birth_mixture.M[0], np.ones(sdim, dtype=dtype))
    assert np.allclose(tracker.birth_mixture.C[0], np.eye(sdim, dtype=dtype))

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("alpha", [0., .5, 1.])
@pytest.mark.parametrize("sparsify", [True, False])
@pytest.mark.parametrize("max_per_row", [25])
def test_tracker_update_and_tracks(sdim, mdim, dtype, alpha, sparsify, max_per_row):

    rng = np.random.default_rng(42)
    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    tracker = MombTracker(birth, F, Q, H, R, 0.95, 0.99, 1e-4, alpha=alpha, dtype=dtype,
                          sparsify=sparsify, max_per_row=max_per_row)
    
    for _ in range(20):
        
        n_meas = rng.integers(0, 40)
        if n_meas > 0:
            Z = rng.random((n_meas, mdim)).astype(dtype)
        else:
            Z = np.zeros((0, mdim), dtype=dtype)
        tracker.step(Z)

        means, covs, ids = tracker.confirmed_tracks()
    
        assert type(means) is np.ndarray
        assert type(covs) is np.ndarray
        assert type(ids) is np.ndarray
        
        assert means.shape[1] == sdim
        assert covs.shape[1:] == (sdim, sdim)

# --------------------------------  IMOMB  ----------------------------------------

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tracker_init(sdim, mdim, dtype):

    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    
    tracker = MombTracker(
        birth_mixture=birth,
        F=F, Q=Q, H=H, R=R,
        PD=0.95, PS=0.99, clutter_int=1e-3,
        PG=0.99, alpha=0.5, min_global_hypot_weight=1e-5, min_bernoulli_exist_prob=1e-5, min_poisson_weight=1e-5,
        max_hypothesis=200, conf_thr=0.6,
        sparsify=True, max_per_row=25, birth_mixture0 = birth, dtype=dtype
    )
    assert tracker is not None
    assert tracker.poiss.size() == 0
    assert tracker.alpha == 0.5
    tracker.alpha = 0.25
    assert tracker.alpha == 0.25

@pytest.mark.parametrize("sdim", SDIMS)
@pytest.mark.parametrize("mdim", MDIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sparsify", [True, False])
@pytest.mark.parametrize("max_per_row", [25])
def test_tracker_update_and_tracks(sdim, mdim, dtype, sparsify, max_per_row):

    rng = np.random.default_rng(42)
    birth, F, Q, H, R = get_tracker_inputs(sdim, mdim, dtype)
    tracker = MombTracker(birth, F, Q, H, R, 0.95, 0.99, 1e-4, 1e-4, alpha=0.5, dtype=dtype,
                          sparsify=sparsify, max_per_row=max_per_row)
    
    for _ in range(20):
        
        n_meas = rng.integers(0, 40)
        if n_meas > 0:
            Z = rng.random((n_meas, mdim)).astype(dtype)
        else:
            Z = np.zeros((0, mdim), dtype=dtype)
        tracker.step(Z)

        means, covs, ids = tracker.confirmed_tracks()
    
        assert type(means) is np.ndarray
        assert type(covs) is np.ndarray
        assert type(ids) is np.ndarray
        
        assert means.shape[1] == sdim
        assert covs.shape[1:] == (sdim, sdim)
