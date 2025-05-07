import numpy as np

from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent
from pymanopt.function import numpy as pymanopt_numpy
from pymanopt.tools import diagnostics

def disentangle(X, dis_dims, svd_dims, max_time=1e100, n_iter=100, initial="identity", algorithm="alternating"):
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

    Xshape = X.shape

    # initial disentangler
    Q = np.eye(np.prod([X.shape[d] for d in dis_dims]))
    X_dis = ten_to_mat(X, dis_dims)
    X_dis = Q @ X_dis
    X = mat_to_ten(X_dis, Xshape, dis_dims)

    X_svd = ten_to_mat(X, svd_dims)
    _, s0, _ = np.linalg.svd(X_svd)

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

if __name__ == "__main__":
    X = np.random.rand(2,3,4,5,6)
    dis_dims = [0, 3]
    svd_dims = [0, 3]

    disentangle(X, dis_dims, svd_dims)