import numpy as np
import matplotlib.pyplot as plt
from disentangle import *


# Define a tensor
X = np.random.rand(8,8,8,8)

dis_legs = [0, 1]
svd_legs = [0, 2]

Qr, Ur, Sr, Vr, logr = disentangle(X, dis_legs, svd_legs, 
                  optimizer="rCG",
                  objective=trunc_error,
                  chi=20,
                  verbosity=1,
                  return_log=True,
                  check_hess=True
                  )