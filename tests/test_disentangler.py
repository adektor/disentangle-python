import numpy as np
import pytest
from tensor_disentangler.disentangle import *

@pytest.fixture
def small_tensor():
    np.random.seed(0)
    return np.random.rand(4, 3, 5, 2)

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

def test_recon(small_tensor):
    X = small_tensor
    dis_legs = [0, 1]
    svd_legs = [0, 2]

    Q, U, S, V, log = disentangle(
        X, dis_legs, svd_legs,
        optimizer="rCG",
        objective=renyi,
        alpha=0.5,
        max_iterations=10,
        return_log=True,
    )
    QX_svd = U @ np.diag(S) @ V
    QX = mat_to_ten(QX_svd, X.shape, svd_legs)
    QX_dis = ten_to_mat(QX, dis_legs)
    X_dis = np.conj(Q.T) @ QX_dis
    X_recon = mat_to_ten(X_dis, X.shape, dis_legs)
    err = np.linalg.norm(X_recon - X)
    assert err < 1e-8

def test_binary_search_tolerance(small_tensor):
    X = small_tensor
    dis_legs = [0, 1]
    svd_legs = [0, 2]
    tol = 1e-6

    Q, U, S, V, chi = disentangle_bs(
        X, dis_legs, svd_legs, tol,
        max_dis=100,
        max_iterations=200,
        verbosity=0,
    )

    U, S, V = U[:, :chi], S[:chi], V[:chi, :]
    QX_svd = U @ np.diag(S) @ V
    QX = mat_to_ten(QX_svd, X.shape, svd_legs)
    QX_dis = ten_to_mat(QX, dis_legs)
    X_dis = np.conj(Q.T) @ QX_dis
    X_recon = mat_to_ten(X_dis, X.shape, dis_legs)
    rel_err = np.linalg.norm(X_recon - X) / np.linalg.norm(X)
    assert rel_err < tol