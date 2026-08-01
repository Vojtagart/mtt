import pytest
import numpy as np
from mtt import GaussianMixture


DIMENSIONS = [2, 3, 4, 5]
DTYPES = [np.float64]

def get_random_gaussian(dim, dtype, rng):
    mu = rng.random(dim).astype(dtype)

    A = rng.random((dim, dim)).astype(dtype)
    cov = np.dot(A, A.T) + np.eye(dim, dtype=dtype) * 1e-3
    
    return mu, cov

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_push(dim, dtype):

    rng = np.random.default_rng(seed=42)

    gm = GaussianMixture(dim, dtype=dtype)
    assert gm.size() == 0, f"Size mismatch. Expected 0, got {gm.size()}"
    cnt = 0

    W = []
    M = []
    C = []

    gm.reserve(10)

    for _ in range(50):

        mu, cov = get_random_gaussian(dim, dtype, rng)
        w = rng.random()

        W.append(w)
        M.append(mu)
        C.append(cov)
    
        gm.push(w, mu, cov)
        cnt += 1
        assert gm.size() == cnt, f"Size mismatch. Expected {cnt}, got {gm.size()}"
    
        assert np.allclose(gm.W, W)
        assert np.allclose(gm.M, M)
        assert np.allclose(gm.C, C)
    
    gm.clear()
    assert gm.size() == 0, f"Size mismatch. Expected 0, got {gm.size()}"

    assert gm.W.size == 0
    assert gm.M.size == 0
    assert gm.C.size == 0

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mutable_properties(dim, dtype):

    gm = GaussianMixture(dim, dtype=dtype)

    z = np.zeros(dim, dtype=dtype)
    I = np.eye(dim, dtype=dtype)
    
    gm.push(0.1, z, I)
    gm.push(0.2, z, I)
    gm.push(0.8, z, I)
    gm.push(0.5, z, I)
    gm.push(0.05, z, I)
    
    view = gm.W
    view[0] = 0.99
    view[3] = 0.64
    
    assert np.isclose(gm.W[0], 0.99)
    assert np.isclose(gm.W[3], 0.64)
    
    view = gm.M
    mu1 = np.full(dim, 5., dtype=dtype)
    mu2 = np.full(dim, 3., dtype=dtype)
    view[1] = mu1
    view[2] = mu2
    
    assert np.allclose(gm.M[1], mu1)
    assert np.allclose(gm.M[2], mu2)

    view = gm.C
    cov1 = np.eye(dim, dtype=dtype) * 2.0
    cov2 = np.eye(dim, dtype=dtype) * 5.0 + np.ones((dim, dim), dtype=dtype)
    view[0] = cov1
    view[4] = cov2
    
    assert np.allclose(gm.C[0], cov1)
    assert np.allclose(gm.C[4], cov2)

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_property_setters(dim, dtype):

    rng = np.random.default_rng(seed=42)

    gm = GaussianMixture(dim, dtype=dtype)
    n_comps = 10
    gm.reserve(n_comps)
    
    for _ in range(n_comps):
        gm.push(0.0, np.zeros(dim, dtype=dtype), np.eye(dim, dtype=dtype))
        
    W = rng.random(n_comps).astype(dtype).ravel()
    M = rng.random(n_comps * dim).astype(dtype).reshape(n_comps, dim)
    C = rng.random(n_comps * dim * dim).astype(dtype).reshape(n_comps, dim, dim)
    for i in range(n_comps):
        C[i] = np.dot(C[i], C[i].T) + np.eye(dim, dtype=dtype) * 1e-3
    
    gm.W = W
    gm.M = M
    gm.C = C
    
    assert np.allclose(gm.W, W)
    assert np.allclose(gm.M, M)
    assert np.allclose(gm.C, C)

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_scale_filter(dim, dtype):

    gm = GaussianMixture(dim, dtype=dtype)

    z = np.zeros(dim, dtype=dtype)
    I = np.eye(dim, dtype=dtype)
    
    gm.push(0.1, z, I)
    gm.push(0.5, z, I)
    gm.push(0.05, z, I)
    
    assert gm.size() == 3
    
    gm.scale_weight(2.0)
    assert np.allclose(gm.W, [0.2, 1.0, 0.1])
    
    gm.filter_out(0.15)

    assert gm.size() == 2
    assert np.allclose(np.sort(gm.W), [0.2, 1.0])
    
    gm.clear()
    assert gm.size() == 0
    assert gm.capacity() > 0 

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_erase(dim, dtype):

    gm = GaussianMixture(dim, dtype=dtype)
    n_comps = 10

    o = np.zeros(dim, dtype=dtype)
    I = np.eye(dim, dtype=dtype)
    
    for i in range(n_comps):
        gm.push(i, o * i, I * i)
    
    gm.erase(4)
    assert gm.size() == n_comps - 1
    
    gm.clear()
    gm.push(1., o, I)
    gm.push(2., 2 * o, 2 * I)
    gm.erase(0)

    assert np.isclose(gm.W[0], 2)
    assert np.allclose(gm.M[0], 2 * o)
    assert np.allclose(gm.C[0], 2 * I)

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moment_matching(dim, dtype):

    gm = GaussianMixture(dim, dtype=dtype)
    
    mu1 = np.zeros(dim, dtype=dtype)
    mu2 = np.zeros(dim, dtype=dtype)
    mu2[0] = 2.0
    
    cov = np.eye(dim, dtype=dtype)
    
    gm.push(0.5, mu1, cov)
    gm.push(0.5, mu2, cov)
    
    res_mu, res_cov = gm.as_gaussian()
    
    actual_mu = np.zeros(dim, dtype=dtype)
    actual_mu[0] = 1.0
    
    actual_cov = np.eye(dim, dtype=dtype)
    actual_cov[0, 0] = 2.0
    
    tol = 1e-2 if dtype == np.float32 else 1e-6
    
    assert np.allclose(res_mu, actual_mu, atol=tol)
    assert np.allclose(res_cov, actual_cov, atol=tol)

def test_error_handling():
    with pytest.raises(ValueError):
        GaussianMixture(0)

    with pytest.raises(ValueError):
        gm = GaussianMixture(2)
        gm.push(-5, np.zeros(2, dtype=float), np.eye(2, dtype=float))

    with pytest.raises(ValueError):
        gm = GaussianMixture(2)
        gm.push(1., np.zeros(3, dtype=float), np.eye(2, dtype=float))

    with pytest.raises(ValueError):
        gm = GaussianMixture(2)
        gm.push(1., np.zeros(2, dtype=float), np.eye(3, dtype=float))

    with pytest.raises(ValueError):
        gm = GaussianMixture(2)
        gm.push(1., np.zeros(3, dtype=float), np.eye(3, dtype=float))
