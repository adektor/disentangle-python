import numpy as np
import matplotlib.pyplot as plt
from disentangle import *

# Define a tensor
X = np.random.rand(8,8,8,8)

dis_legs = [0, 1]
svd_legs = [0, 2]

Qeye = np.eye(np.prod([X.shape[d] for d in dis_legs])) 
U0, S0, V0 = disentangled_usv(X, Qeye, dis_legs, svd_legs)

Qr, Ur, Sr, Vr, logr = disentangle(X, dis_legs, svd_legs, 
                  optimizer="cg",
                  objective=nuclear,
                  alpha=1/2,
                  verbosity=1,
                  return_log=True,
                  check_grad=True,
                  check_hess=True
                  )

# plot results
plt.figure()
plt.semilogy(S0, label="Q=I")
plt.semilogy(Sr, label="Riemannian opt")
plt.ylabel('singular values')
plt.xlabel('index')
plt.legend()
plt.show()