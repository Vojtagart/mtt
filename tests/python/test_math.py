import pytest
import numpy as np
from scipy.stats import multivariate_normal
from mtt import mvn_logpdf, mahalanobis_distance


DIMENSIONS = [2, 3, 4, 5]
DTYPES = [np.float64, np.float32]

def get_random_gaussian(dim, dtype, rng):
    mu = rng.random(dim).astype(dtype)

    A = rng.random((dim, dim)).astype(dtype)
    cov = np.dot(A, A.T) + np.eye(dim, dtype=dtype) * 1e-3
    
    return mu, cov

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mvn_logpdf(dim, dtype):

    rng = np.random.default_rng(seed=42)

    for _ in range(100):

        mu, cov = get_random_gaussian(dim, dtype, rng)
        x = rng.random(dim).astype(dtype)
        
        mtt_res = mvn_logpdf(x, mu, cov)
        actual_res = multivariate_normal.logpdf(x, mean=mu, cov=cov)
        
        atol = 1 if dtype == np.float32 else 1e-8
        assert np.isclose(mtt_res, actual_res, atol=atol), f"Returned pdf {mtt_res} != scipy pdf {actual_res}"

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mahalanobis_distance(dim, dtype):

    rng = np.random.default_rng(seed=42)

    for _ in range(100):

        mu, cov = get_random_gaussian(dim, dtype, rng)
        x = rng.random(dim).astype(dtype)
    
        mtt_res = mahalanobis_distance(x, mu, cov)
    
        diff = x - mu
        inv = np.linalg.inv(cov)
        actual_res = diff.T @ inv @ diff

        atol = 1 if dtype == np.float32 else 1e-6
        assert np.isclose(mtt_res, actual_res, atol=atol), f"Returned Mahal. dist. {mtt_res} != numpy Mahal. dist. {actual_res}"

def test_error_handling():
    
    dim = 2
    dtype = np.float64
    rng = np.random.default_rng(seed=42)

    mu, cov = get_random_gaussian(dim, dtype, rng)
    x = rng.random(dim).astype(dtype)
    
    with pytest.raises(ValueError):
        mvn_logpdf(np.zeros(3), mu, cov)
    with pytest.raises(ValueError):
        mvn_logpdf(x, np.zeros(3), cov)
    with pytest.raises(ValueError):
        mvn_logpdf(x, mu, np.zeros((3, 3)))

    with pytest.raises(ValueError):
        mahalanobis_distance(np.zeros(3), mu, cov)
    with pytest.raises(ValueError):
        mahalanobis_distance(x, np.zeros(3), cov)
    with pytest.raises(ValueError):
        mahalanobis_distance(x, mu, np.zeros((3, 3)))
