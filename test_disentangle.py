import numpy as np
import matplotlib.pyplot as plt
from disentangle import *

X = np.random.rand(12,12,12,12)
dis_dims = [0, 1]
svd_dims = [0, 2]

X_svd = ten_to_mat(X, svd_dims)
_, s0, _ = np.linalg.svd(X_svd)

# Alternating:
Q = disentangle(X, dis_dims, svd_dims, chi = 12, n_iter = 1000, verbose=True)

# Riemannian:
# Q = disentangle(X, dis_dims, svd_dims, algorithm="Riemannian")

QX_dis = Q @ ten_to_mat(X, dis_dims)
QX = mat_to_ten(QX_dis, X.shape, dis_dims)
QX_svd = ten_to_mat(QX, svd_dims)
_, s, _ = np.linalg.svd(QX_svd)
plt.semilogy(s0, label="before")
plt.semilogy(s, label="after")
plt.legend()
plt.show()