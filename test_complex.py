import numpy as np
import matplotlib.pyplot as plt
from disentangle import *

# Define a tensor
# X = np.random.rand(8,8,8,8) 
X = np.random.rand(8,8,8,8) + 1j*np.random.rand(8,8,8,8)

dis_legs = [0, 1]
svd_legs = [0, 2]

# Reshape to matrix of shape (l*r, b*c)
B = ten_to_mat(X, dis_legs)
print(B)

Bsvd = ten_to_mat(X, svd_legs)
print(Bsvd)

Qeye = np.eye(np.prod([X.shape[d] for d in dis_legs]))
U0, S0, V0 = disentangled_usv(X, Qeye, dis_legs, svd_legs)

Qr, Ur, Sr, Vr, logr = disentangle(X, dis_legs, svd_legs, 
                  optimizer="cg",
                  objective=trunc_error,
                  min_grad_norm=1e-12,
                  max_iterations=1500,
                  chi=32,
                  alpha=0.5,
                  verbosity=1,
                  return_log=True,
                  check_grad=True,
                  check_hess=True
                  )

# plot results
plt.figure()
plt.semilogy(S0, label="Q=I")
plt.semilogy(Sr, label="optimized")
plt.ylabel('singular values')
plt.xlabel('index')
plt.legend()

plt.figure()
plt.semilogy(logr["cost_history"])
plt.title('Riemannian optimizer cost')
plt.xlabel('iteration')
plt.show()