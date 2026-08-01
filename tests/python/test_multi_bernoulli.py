import pytest
import numpy as np
from mtt import MultiBernoulli


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

    bm = MultiBernoulli(dim, dtype=dtype)
    assert bm.size() == 0, f"Size mismatch. Expected 0, got {bm.size()}"
    cnt = 0

    R = []
    M = []
    C = []

    bm.reserve(10)

    for _ in range(50):

        mu, cov = get_random_gaussian(dim, dtype, rng)
        r = rng.random()

        R.append(r)
        M.append(mu)
        C.append(cov)
    
        bm.push(r, mu, cov)
        cnt += 1
        assert bm.size() == cnt, f"Size mismatch. Expected {cnt}, got {bm.size()}"

        assert np.allclose(bm.R, R)
        assert np.allclose(bm.M, M)
        assert np.allclose(bm.C, C)
    
    bm.clear()
    assert bm.size() == 0, f"Size mismatch. Expected 0, got {bm.size()}"

    assert bm.R.size == 0
    assert bm.M.size == 0
    assert bm.C.size == 0

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mutable_properties(dim, dtype):

    bm = MultiBernoulli(dim, dtype=dtype)

    z = np.zeros(dim, dtype=dtype)
    I = np.eye(dim, dtype=dtype)
    
    bm.push(0.5, z, I)
    bm.push(0.9, z, I)
    bm.push(0.6, z, I)
    bm.push(0.1, z, I)
    bm.push(0.2, z, I)

    view = bm.R
    view[0] = 0.99
    view[3] = 0.64
    
    assert np.isclose(bm.R[0], 0.99)
    assert np.isclose(bm.R[3], 0.64)
    
    view = bm.M
    mu1 = np.full(dim, 5., dtype=dtype)
    mu2 = np.full(dim, 3., dtype=dtype)
    view[1] = mu1
    view[2] = mu2
    
    assert np.allclose(bm.M[1], mu1)
    assert np.allclose(bm.M[2], mu2)

    view = bm.C
    cov1 = np.eye(dim, dtype=dtype) * 2.0
    cov2 = np.eye(dim, dtype=dtype) * 5.0 + np.ones((dim, dim), dtype=dtype)
    view[0] = cov1
    view[4] = cov2
    
    assert np.allclose(bm.C[0], cov1)
    assert np.allclose(bm.C[4], cov2)

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_property_setters(dim, dtype):

    rng = np.random.default_rng(seed=42)

    bm = MultiBernoulli(dim, dtype=dtype)
    n_comps = 10
    bm.reserve(n_comps)
    
    for _ in range(n_comps):
        bm.push(0.0, np.zeros(dim, dtype=dtype), np.eye(dim, dtype=dtype))
    
    R = rng.random(n_comps).astype(dtype).ravel()
    M = rng.random(n_comps * dim).astype(dtype).reshape(n_comps, dim)
    C = rng.random(n_comps * dim * dim).astype(dtype).reshape(n_comps, dim, dim)
    for i in range(n_comps):
        C[i] = np.dot(C[i], C[i].T) + np.eye(dim, dtype=dtype) * 1e-3
    
    bm.R = R
    bm.M = M
    bm.C = C
    
    assert np.allclose(bm.R, R)
    assert np.allclose(bm.M, M)
    assert np.allclose(bm.C, C)

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_scale_filter(dim, dtype):

    bm = MultiBernoulli(dim, dtype=dtype)

    z = np.zeros(dim, dtype=dtype)
    I = np.eye(dim, dtype=dtype)
    
    bm.push(0.2, z, I)
    bm.push(0.3, z, I)
    bm.push(0.1, z, I)
    
    assert bm.size() == 3

    bm.scale_exist_prob(2.0)
    assert np.allclose(bm.R, [0.4, 0.6, 0.2])

    bm.filter_out(0.5)

    assert bm.size() == 1
    assert np.allclose(bm.R, [0.6])
    
    bm.clear()
    assert bm.size() == 0
    assert bm.capacity() > 0 

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_erase(dim, dtype):

    bm = MultiBernoulli(dim, dtype=dtype)
    n_comps = 10

    o = np.zeros(dim, dtype=dtype)
    I = np.eye(dim, dtype=dtype)
    
    for i in range(n_comps):
        bm.push(0.5, o * i, I * i)
    
    bm.erase(4)
    assert bm.size() == n_comps - 1
    
    bm.clear()
    bm.push(0.5, o, I)
    bm.push(0.7, 2 * o, 2 * I)
    bm.erase(0)

    assert np.isclose(bm.R[0], 0.7)
    assert np.allclose(bm.M[0], 2 * o)
    assert np.allclose(bm.C[0], 2 * I)

def test_error_handling():
    with pytest.raises(ValueError):
        MultiBernoulli(0)

    with pytest.raises(ValueError):
        bm = MultiBernoulli(2)
        bm.push(1.2, np.zeros(2, dtype=float), np.eye(2, dtype=float))

    with pytest.raises(ValueError):
        bm = MultiBernoulli(2)
        bm.push(0.5, np.zeros(3, dtype=float), np.eye(2, dtype=float))

    with pytest.raises(ValueError):
        bm = MultiBernoulli(2)
        bm.push(0.5, np.zeros(2, dtype=float), np.eye(3, dtype=float))

    with pytest.raises(ValueError):
        bm = MultiBernoulli(2)
        bm.push(0.5, np.zeros(3, dtype=float), np.eye(3, dtype=float))
