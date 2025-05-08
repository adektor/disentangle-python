import numpy as np
from disentangle import *

X = np.random.rand(4,5,6,7)
dis_dims = [0, 1]
svd_dims = [0, 2]

X_svd = ten_to_mat(X, svd_dims)
_, s0, _ = np.linalg.svd(X_svd)

Q = disentangle(X, dis_dims, svd_dims)

QX_dis = Q @ ten_to_mat(X, dis_dims)
QX = mat_to_ten(QX_dis, X.shape, dis_dims)
QX_svd = ten_to_mat(QX, svd_dims)
_, s, _ = np.linalg.svd(QX_svd)

# print(Q)
# print(s-s0)
plt.semilogy(s0, label="before")
plt.semilogy(s, label="after")
plt.legend()
plt.show()