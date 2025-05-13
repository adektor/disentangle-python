import numpy as np
import pymanopt
from rank_surrogates import *

from pymanopt.tools import diagnostics # for gradient and hessian checks

def disentangle(X, dis_legs, svd_legs,
                chi=20,
                n_iter=300,
                initial="identity",
                algorithm="alternating",
                surrogate=renyi,
                dcost_thresh=1e-10,
                max_time=1e100,
                verbose=False):
    '''
    Optimize a unitary matrix Q that contracts with dis_legs of X
    to minimize the entanglement across matrix with rows indexed by svd_legs. 

    X: numpy array with 0,1,...,n-1 dimensions
    dis_legs: list of dimensions indicating legs the disentangler is applied to
    svd_legs: list of dimensions indicating legs for disentangling
    Q0: initial disentangler (unitary matrix of size () )
    max_time: max wall time
    initial: initialization of disentangler
    algorithm: disentangler algorithm ()
    '''

    assert all(0 <= d < X.ndim for d in dis_legs), "Invalid dimension in dis_legs"
    assert all(0 <= d < X.ndim for d in svd_legs), "Invalid dimension in svd_legs"
    dis_legs = sorted(set(dis_legs))
    svd_legs = sorted(set(svd_legs))

    n = np.prod([X.shape[d] for d in dis_legs]) # disentangler is n x n
    
    X_svd = ten_to_mat(X, svd_legs)
    s0 = np.linalg.svd(X_svd, compute_uv=False)

    if initial == "identity":
        Q0 = np.eye(n)
    else:
        raise ValueError("initial option not supported")

    if algorithm == "alternating":
        Q = Q0
        cost = [np.linalg.norm(s0[chi:])]
        for i in range(n_iter):
            X_dis = ten_to_mat(X, dis_legs)
            QX_dis = Q @ X_dis
            QX = mat_to_ten(QX_dis, X.shape, dis_legs)
            QX_svd = ten_to_mat(QX, svd_legs)

            u, s, v = np.linalg.svd(QX_svd)
            cost.append(np.linalg.norm(s[chi:]))
            u, s, v = u[:,:chi], s[:chi], v[:chi,:]

            QX_svd_chi = u@np.diag(s)@v
            QX_chi = mat_to_ten(QX_svd_chi, X.shape, svd_legs)

            M = ten_to_mat(QX_chi, dis_legs) @ (X_dis.T)
            u, _, v = np.linalg.svd(M, full_matrices=False)
            Q = u@v
            
            if i>0 and np.abs(cost[-1]-cost[-2]) < dcost_thresh:
                if verbose:
                    print("exiting at iteration {0}".format(i))
                break

        if verbose:
            print("reduced truncation error from {0} to {1} "
                    "in {2} iterations of alternating".format(cost[0], cost[-1], i))
            

    elif algorithm == "Riemannian":
        phi = surrogate
        manifold = pymanopt.manifolds.Stiefel(n, n)

        @pymanopt.function.numpy(manifold)
        def cost(Q):
            X_dis = ten_to_mat(X, dis_legs)
            QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
            QX_svd = ten_to_mat(QX, svd_legs)
            u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

            phi, _, _ = surrogate(s[chi:])
            cost = np.sum(phi)
            # cost = np.linalg.norm(s[chi:])**2
            # cost = np.linalg.norm(s)**2
            return cost
        
        @pymanopt.function.numpy(manifold)
        def egrad(Q):
            # The following 4 lines are also computed in cost... 
            X_dis = ten_to_mat(X, dis_legs)
            QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
            QX_svd = ten_to_mat(QX, svd_legs)
            u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

            # Euclidian gradient
            _, d_phi, _ = surrogate(s[chi:])
            d_phi = np.hstack([np.zeros(chi), d_phi])
            egrad = ten_to_mat(mat_to_ten(u@np.diag(d_phi)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T)
    
            return egrad
        
        @pymanopt.function.numpy(manifold)
        def ehess(Q, E):
            # The following 4 lines are also computed in cost & egrad... 
            X_dis = ten_to_mat(X, dis_legs)
            QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
            QX_svd = ten_to_mat(QX, svd_legs)
            u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

            e_grad = egrad(Q)
            # rgrad = manifold.projection(Q, e_grad)

            # F matrix (4.27)
            s_diff_matrix = (s**2)[:, None] - (s**2)[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                f = 1.0 / s_diff_matrix
                np.fill_diagonal(f, 0.0)

            m, k, n = u.shape[0], u.shape[1], v.shape[1]
            EX_dis = E@ten_to_mat(X, dis_legs)
            EX_svd = ten_to_mat(mat_to_ten(EX_dis, X.shape, dis_legs), svd_legs)

            DUE = u@(f*(u.T@EX_svd@v.T@np.diag(s) + np.diag(s)@v@EX_svd.T@u)) + (np.eye(m) - u@u.T)@EX_svd@v.T@np.diag(1/s)
            DVE = v.T@(f*(np.diag(s)@u.T@EX_svd@v.T + v@EX_svd.T@u@np.diag(s))) + (np.eye(n) - v.T@v)@EX_svd.T@u@np.diag(1/s)

            _, d_phi, dd_phi = surrogate(s[chi:])
            d_phi = np.hstack([np.zeros(chi), d_phi])
            dd_phi = np.hstack([np.zeros(chi), dd_phi])

            DSE = np.diag(dd_phi)@u.T@EX_svd@v.T
            
            Degrad = ( ten_to_mat(mat_to_ten(DUE@np.diag(d_phi)@v, X.shape, svd_legs), dis_legs) 
                         + ten_to_mat(mat_to_ten(u@DSE@v, X.shape, svd_legs), dis_legs) 
                         + ten_to_mat(mat_to_ten(u@np.diag(d_phi)@DVE.T, X.shape, svd_legs), dis_legs) )

            left = Degrad@X_dis.T
            right = E@e_grad.T@Q + Q@left.T@Q + Q@e_grad.T@E
            lr = 0.5*(left - right)
            ehess = manifold.projection(Q, lr)

            return ehess
        
        problem = pymanopt.Problem(manifold=manifold, 
                                   cost=cost, 
                                   euclidean_gradient=egrad,
                                #    euclidean_hessian=ehess
                                   )
        # diagnostics.check_gradient(problem)
        # diagnostics.check_hessian(problem)
        
        solver = pymanopt.optimizers.SteepestDescent(verbosity=0)
        Q = solver.run(problem, initial_point=Q0).point

    else:
        raise ValueError("algorithm option not supported")

    return Q

def ten_to_mat(X, row_legs):
        ''' 
        Reshapes a tensor X into a matrix X_mat with 
        dimensions row_legs of X indexing the rows of X_mat. 

        X: numpy array
        row_legs: list of dimensions of X
        '''

        all_legs = list(range(X.ndim))
        col_legs = [d for d in all_legs if d not in row_legs]
        perm = row_legs + col_legs
        X_perm = X.transpose(perm)

        row_size = np.prod([X.shape[d] for d in row_legs])
        col_size = np.prod([X.shape[d] for d in col_legs])

        X_mat = X_perm.reshape(row_size, col_size)
        return X_mat

def mat_to_ten(X_mat, orig_shape, row_legs):
    ''' 
    Reconstructs a tensor from its matrix form X_mat.

    X_mat: 2D numpy array (matrix)
    orig_shape: original shape of the tensor before flattening
    row_legs: list of dimensions that were used as rows in the matrix
    '''

    # Validate input
    N = len(orig_shape)
    assert X_mat.ndim == 2
    assert all(0 <= d < N for d in row_legs)
    
    # Compute complement dimensions (col_legs)
    all_legs = list(range(N))
    col_legs = [d for d in all_legs if d not in row_legs]
    
    # Get sizes of row and column dimensions
    row_shape = [orig_shape[d] for d in row_legs]
    col_shape = [orig_shape[d] for d in col_legs]

    # Reshape into full tensor with permuted dimensions
    full_shape = row_shape + col_shape
    X_perm = X_mat.reshape(full_shape)

    # Invert permutation
    perm = row_legs + col_legs
    inv_perm = np.argsort(perm)
    X = X_perm.transpose(inv_perm)

    return X