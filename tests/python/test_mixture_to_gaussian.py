import pytest
import numpy as np
from mtt import GaussianMixture, MixtureToGaussian

DIMENSIONS = [2, 3, 4, 5]
DTYPES = [np.float64]

def get_random_gaussian(dim, dtype, rng):
    mu = rng.random(dim).astype(dtype)

    A = rng.random((dim, dim)).astype(dtype)
    cov = np.dot(A, A.T) + np.eye(dim, dtype=dtype) * 1e-3
    
    return mu, cov


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_correctness(dim, dtype):

    rng = np.random.default_rng(seed=42)
    
    gm = GaussianMixture(dim, dtype=dtype)
    mtg = MixtureToGaussian(dim, dtype=dtype)
    
    n_comps = 25
    
    for _ in range(n_comps):
        w = rng.random()
        mu, cov = get_random_gaussian(dim, dtype, rng)
        
        gm.push(w, mu, cov)
        mtg.add_gauss(w, mu, cov)

    gm_mu, gm_cov = gm.as_gaussian()
    mtg_mu, mtg_cov = mtg.get_gauss()
    
    tol = 1e-2 if dtype == np.float32 else 1e-6
    assert np.allclose(mtg_mu, gm_mu, atol=tol), "Means do not match"
    assert np.allclose(mtg_cov, gm_cov, atol=tol), "Covariances do not match"


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_edge_cases(dim, dtype):
    mtg = MixtureToGaussian(dim, dtype=dtype)
    
    # Test it do not crash
    _, _ = mtg.get_gauss()
    
    z = np.zeros(dim, dtype=dtype)
    I = np.eye(dim, dtype=dtype)
    mtg.add_gauss(0.0, z, I)
    mtg.add_gauss(0.0, z, I)
    mtg.add_gauss(0.0, z, I)
    
    # Test it do not crash
    _, _ = mtg.get_gauss()

def test_init_error():
    with pytest.raises(ValueError):
        MixtureToGaussian(0)

    with pytest.raises(ValueError):
        mtg = MixtureToGaussian(2)
        mtg.add_gauss(-0.2, np.zeros(2, dtype=float), np.eye(2, dtype=float))

    with pytest.raises(ValueError):
        mtg = MixtureToGaussian(2)
        mtg.add_gauss(1., np.zeros(3, dtype=float), np.eye(2, dtype=float))

    with pytest.raises(ValueError):
        mtg = MixtureToGaussian(2)
        mtg.add_gauss(1., np.zeros(2, dtype=float), np.eye(3, dtype=float))

    with pytest.raises(ValueError):
        mtg = MixtureToGaussian(2)
        mtg.add_gauss(1., np.zeros(3, dtype=float), np.eye(3, dtype=float))
