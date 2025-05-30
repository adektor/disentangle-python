import autograd.numpy as np
import pymanopt
from pymanopt.manifolds import UnitaryGroup
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from pymanopt.tools.diagnostics import check_gradient, check_hessian

''' A simple example to test Pymanopt Stiefel and Unitary manifolds. 
    Minimize f(Q) = \| AQ - B \|_F^2
    where A,B are n x n real or complex matrices. 
'''

n = 4
A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
B = np.random.randn(n, n) + 1j * np.random.randn(n, n)
manifold = UnitaryGroup(n)

# --- Cost function ---
@pymanopt.function.numpy(manifold)
def cost(Q):
    diff = A @ Q - B
    return np.real(np.trace(diff.conj().T @ diff))  # Frobenius norm squared

# --- Euclidean gradient ---
@pymanopt.function.numpy(manifold)
def egrad(Q):
    return 2 * A.conj().T @ (A @ Q - B)  # ∇_Q f = 2 A^{\ast}(AQ - B)

# --- Euclidean Hessian ---
@pymanopt.function.numpy(manifold)
def ehess(Q, H):
    return 2 * A.conj().T @ (A @ H)

# --- Set up problem ---
problem = pymanopt.Problem(manifold=manifold, 
                            cost=cost, 
                            euclidean_gradient=egrad,
                            euclidean_hessian=ehess
                            )

check_gradient(problem)
check_hessian(problem)

# --- Solve optimization ---
solver = ConjugateGradient()
result = solver.run(problem)