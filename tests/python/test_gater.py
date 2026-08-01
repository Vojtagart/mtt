import pytest
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from mtt import Gater

DIMENSIONS = [2, 3, 4, 5]
SIZES = [100, 200, 500]
DTYPES = [np.float64]


def get_random_gaussian(dim, dtype, rng):
    mu = rng.random(dim).astype(dtype)

    A = rng.random((dim, dim)).astype(dtype)
    cov = np.dot(A, A.T) + np.eye(dim, dtype=dtype) * 1e-3
    
    return mu, cov

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gater_init(dim, dtype):
    gater = Gater(dim, dtype=dtype)
    assert gater is not None

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_correctness(dim, size, dtype):

    rng = np.random.default_rng(seed=42)

    gater = Gater(dim, dtype=dtype)
    atol = 1e-3 if dtype == np.float32 else 1e-6

    for _ in range(20):

        Z = rng.random(size=dim*size).astype(dtype).reshape(-1, dim)
        gater.set_measurements(Z)

        mean, cov = get_random_gaussian(dim, dtype, rng)
        
        Z = Z - mean.ravel()
        cho = cho_factor(cov, lower=True, check_finite=False)
        sol = cho_solve(cho, Z.T).T
        d = np.einsum("ij,ij->i", Z, sol)
        d = np.clip(d, 0.0, None)

        for _ in range(20):

            thr = rng.random() * 10
    
            idxs = gater.gate(mean, cov, thr)
            
            mask_up = (d <= thr + atol)
            mask_down = (d <= thr - atol)
            up_size, down_size = np.sum(mask_up), np.sum(mask_down)

            assert down_size <= idxs.size <= up_size, f"Size missmatch. Reported {idxs.size}, allowed range [{down_size}, {up_size}]"
            assert mask_down[idxs].sum() == down_size and mask_up[idxs].sum() == idxs.size, f"Reported indexes contain invalid indexes"

@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gater_empty(dim, dtype):

    gater = Gater(dim, dtype=dtype)
    
    gater.set_measurements(np.zeros((0, dim), dtype=dtype))
    
    idxs = gater.gate(np.zeros(dim, dtype=dtype), np.eye(dim, dtype=dtype), 10.0)
    assert len(idxs) == 0
    assert idxs.dtype == np.uint32 or idxs.dtype == np.int32

def test_error_handling():

    gater = Gater(2, dtype=float)

    with pytest.raises(ValueError):
        gater.set_measurements(np.zeros((5, 3), dtype=float))

    gater.set_measurements(np.zeros((5, 2), dtype=float))

    with pytest.raises(ValueError):
        gater.gate(np.zeros(3), np.eye(2), 10)

    with pytest.raises(ValueError):
        gater.gate(np.zeros(2), np.eye(3), 10)

    with pytest.raises(ValueError):
        gater.gate(np.zeros(3), np.eye(3), 10)
