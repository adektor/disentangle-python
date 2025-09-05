import numpy as np
import pytest
from tensor_disentangler.disentangle import disentangle, disentangle_bs, disentangled_usv, renyi, trunc_error

@pytest.fixture
def small_tensor():
    np.random.seed(0)
    return np.random.rand(4, 4, 4, 4)

def test_shapes_and_isometry(small_tensor):
    X = small_tensor
    dis_legs = [0, 1]
    svd_legs = [0, 2]

    Q, U, S, V, log = disentangle(
        X, dis_legs, svd_legs,
        optimizer="rCG",
        objective=renyi,
        alpha=0.5,
        max_iterations=100,
        return_log=True,
    )

    # Q should be square of correct size
    dim = np.prod([X.shape[d] for d in dis_legs])
    assert Q.shape == (dim, dim)


    m = int(np.prod([X.shape[d] for d in svd_legs]))                      # rows for U
    other_legs = [i for i in range(X.ndim) if i not in svd_legs]
    n = int(np.prod([X.shape[d] for d in other_legs]))                    # cols for V
    
    assert U.shape[0] == m
    assert V.shape[1] == n
    assert len(S) <= min(U.shape[1], V.shape[0])

    # Q should be approximately orthogonal
    I = Q @ np.conj(Q.T)
    assert np.allclose(I, np.eye(dim), atol=1e-10)

def test_cost_decreases(small_tensor):
    X = small_tensor
    dis_legs = [0, 1]
    svd_legs = [0, 2]

    _, _, _, _, log = disentangle(
        X, dis_legs, svd_legs,
        optimizer="rCG",
        objective=renyi,
        alpha=0.5,
        max_iterations=10,
        return_log=True,
    )

    costs = log["cost_history"]
    assert costs[-1] <= costs[0]

def test_binary_search_tolerance(small_tensor):
    X = small_tensor
    dis_legs = [0, 1]
    svd_legs = [0, 2]

    Q, U, S, V = disentangle_bs(
        X, dis_legs, svd_legs, tol=1e-3,
        max_dis=100,
        max_iterations=200,
        verbosity=0,
    )

    # Check relative error of returned factorization
    rel_err = np.linalg.norm(S[len(S)//2:]) / np.linalg.norm(S)
    assert rel_err < 1e-3