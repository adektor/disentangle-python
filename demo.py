import numpy as np
import matplotlib.pyplot as plt
from disentangle import *

# Define a tensor
X = np.random.rand(12,12,12,12)

# We will apply the disentangler to dimensions (or legs) 0,1 of X
dis_legs = [0, 1]

# The rows of the matrix we SVD will be indexed by dimensions (or legs) 0,2 of X
svd_legs = [0, 2]

# A non-optimal disentangler is the identity matrix of appropriate dimension
Qeye = np.eye(np.prod([X.shape[d] for d in dis_legs])) 

# Let's use this as a benchmark to compare with the optimized disentanglers
U0, S0, V0 = disentangled_usv(X, Qeye, dis_legs, svd_legs)

# Optimize a disentangler Q by minimizing Renyi-1/2 entropy with Riemannian-CG:
Qr, Ur, Sr, Vr = disentangle(X, dis_legs, svd_legs, 
                  optimizer="rCG", 
                  objective=renyi,
                  alpha=0.5,
                  )

# Let's compare with a second disentangler optimized for rank-30 truncation error using the alternating optimizer
Qa, Ua, Sa, Va = disentangle(X, dis_legs, svd_legs, 
                  optimizer="alternating", 
                  objective=trunc_error,
                  chi=60,
                  )

# plot results
plt.figure()
plt.semilogy(S0, label="Q=I")
plt.semilogy(Sr, label="Renyi-CG")
plt.semilogy(Sa, label="trunc-alt")
plt.ylabel('singular values')
plt.xlabel('index')
plt.legend()
plt.show()