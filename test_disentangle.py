import numpy as np
import matplotlib.pyplot as plt
from disentangle import *

X = np.random.rand(12,12,12,12)
dis_legs = [0, 1]
svd_legs = [0, 2]
I = np.eye(np.prod([X.shape[d] for d in dis_legs])) 

def disentangled_spectrum(X, Q, dis_legs, svd_legs):
    QX_dis = Q @ ten_to_mat(X, dis_legs)
    QX = mat_to_ten(QX_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    _, s, _ = np.linalg.svd(QX_svd)
    return s

# No disentangler
s0 = disentangled_spectrum(X, I, dis_legs, svd_legs)

# Alternating optimizer:
# Q_alt = disentangle(X, dis_legs, svd_legs, chi = 12, n_iter = 1000, verbose=True)
# s_alt = disentangled_spectrum(X, Q_alt, dis_legs, svd_legs)

# Riemannian:
Q_r = disentangle(X, dis_legs, svd_legs, algorithm="Riemannian", chi=0, check_grad=True)
s_r = disentangled_spectrum(X, Q_r, dis_legs, svd_legs)

# plot results
plt.figure()
plt.semilogy(s0, label="no disentangler")
# plt.semilogy(s_alt, label="alternating")
plt.semilogy(s_r, label="Riemannian")
plt.ylabel('singular values')
plt.xlabel('index')
plt.legend()
plt.show()