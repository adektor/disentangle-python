import numpy as np
import matplotlib.pyplot as plt
import pymanopt

# from pymanopt import Problem
# from pymanopt.manifolds import Stiefel
# from pymanopt.optimizers import SteepestDescent
# from pymanopt.function import numpy as pymanopt_numpy
# from pymanopt.tools import diagnostics

def disentangle(X, dis_dims, svd_dims, max_time=1e100, chi=20, n_iter=300, initial="identity", algorithm="alternating"):
    '''
    Optimize a unitary matrix Q that contracts with dis_dims of X
    to minimize the entanglement across matrix with rows indexed by svd_dims. 

    X: numpy array with 0,1,...,n-1 dimensions
    dis_dims: list of dimensions indicating legs the disentangler is applied to
    svd_dims: list of dimensions indicating legs for disentangling
    Q0: initial disentangler (unitary matrix of size () )
    max_time: max wall time
    initial: initialization of disentangler
    algorithm: disentangler algorithm ()
    '''

    assert all(0 <= d < X.ndim for d in dis_dims), "Invalid dimension in dis_dims"
    assert all(0 <= d < X.ndim for d in svd_dims), "Invalid dimension in svd_dims"
    dis_dims = sorted(set(dis_dims))
    svd_dims = sorted(set(svd_dims))

    if initial == "identity":
        Q = np.eye(np.prod([X.shape[d] for d in dis_dims]))
    else:
        raise ValueError("initial option not supported")

    if algorithm == "alternating":
        for i in range(n_iter):
            X_dis = ten_to_mat(X, dis_dims)
            QX_dis = Q @ X_dis
            QX = mat_to_ten(QX_dis, X.shape, dis_dims)
            QX_svd = ten_to_mat(QX, svd_dims)

            u, s, v = np.linalg.svd(QX_svd)
            err = np.linalg.norm(s[chi:])
            u, s, v = u[:,:chi], s[:chi], v[:chi,:]

            QX_svd_chi = u@np.diag(s)@v
            QX_chi = mat_to_ten(QX_svd_chi, X.shape, svd_dims)

            M = ten_to_mat(QX_chi, dis_dims) @ (ten_to_mat(X, dis_dims).T)
            u, _, v = np.linalg.svd(M, full_matrices=False)
            Q = u@v

    elif algorithm == "Riemannian":
        manifold = pymanopt.manifolds.Stiefel(A.shape[1], A.shape[0]*A.shape[2])

        # @pymanopt.function.numpy(manifold)
        # def cost(X):
        #     X = X.reshape(A.shape[1], A.shape[0], A.shape[2]).transpose(1, 0, 2)
        #     c = ncon([X, B, C, X, B, C], [(9, 1, 2), (2, 4, 5), (5, 7, 9), (8, 1, 3), (3, 4, 6), (6, 7, 8)]) 
        #     c -= 2*ncon([X, B, C, T], [(6, 1, 2), (2, 3, 4), (4, 5, 6), (1, 3, 5)]) 
        #     c += ncon([T, T], [(1, 2, 3), (1, 2, 3)])
        #     return c/2
        
        # @pymanopt.function.numpy(manifold)
        # def egrad(X):
        #     X = X.reshape(A.shape[1], A.shape[0], A.shape[2]).transpose(1, 0, 2)
        #     g = ncon([B, C, X, B, C], [(-3, 1, 3), (3, 4, -1), (6, -2, 5), (5, 1, 2), (2, 4, 6)]) 
        #     g -= ncon([B, C, T], [(-3, 1, 2), (2, 3, -1), (-2, 1, 3)])
        #     return g.transpose(1, 0, 2).reshape(A.shape[1], A.shape[0]*A.shape[2])
            


    else:
        raise ValueError("algorithm option not supported")

    return Q

def ten_to_mat(X, row_dims):
        ''' 
        Reshapes a tensor X into a matrix X_mat with 
        dimensions row_dims of X indexing the rows of X_mat. 

        X: numpy array
        row_dims: list of dimensions of X
        '''

        all_dims = list(range(X.ndim))
        col_dims = [d for d in all_dims if d not in row_dims]
        perm = row_dims + col_dims
        X_perm = X.transpose(perm)

        row_size = np.prod([X.shape[d] for d in row_dims])
        col_size = np.prod([X.shape[d] for d in col_dims])

        X_mat = X_perm.reshape(row_size, col_size)
        return X_mat

def mat_to_ten(X_mat, orig_shape, row_dims):
    ''' 
    Reconstructs a tensor from its matrix form X_mat.

    X_mat: 2D numpy array (matrix)
    orig_shape: original shape of the tensor before flattening
    row_dims: list of dimensions that were used as rows in the matrix
    '''

    # Validate input
    N = len(orig_shape)
    assert X_mat.ndim == 2
    assert all(0 <= d < N for d in row_dims)
    
    # Compute complement dimensions (col_dims)
    all_dims = list(range(N))
    col_dims = [d for d in all_dims if d not in row_dims]
    
    # Get sizes of row and column dimensions
    row_shape = [orig_shape[d] for d in row_dims]
    col_shape = [orig_shape[d] for d in col_dims]

    # Reshape into full tensor with permuted dimensions
    full_shape = row_shape + col_shape
    X_perm = X_mat.reshape(full_shape)

    # Invert permutation
    perm = row_dims + col_dims
    inv_perm = np.argsort(perm)
    X = X_perm.transpose(inv_perm)

    return X