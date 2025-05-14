import numpy as np
import pymanopt
from utilities import *
from pymanopt.tools import diagnostics

# TODO: - Printing and verbosity in alternating optimizer to be similar to the verbosity of Pymanopt
#       - Check user-supplied objective and optimizer parameters
#       - Return info on the performance of the optimization
#       - Riemannian Hessian(s)

def disentangle(X, dis_legs, svd_legs,
                initial="identity",
                max_iterations=500,
                min_grad_norm=1e-6,
                max_time=1e100,
                optimizer="rCG",
                objective=renyi,
                alpha=0.5,
                chi=0,
                verbosity=0,
                check_grad=False,
                check_hess=False):
    '''
    Optimize a unitary matrix Q that contracts with dis_legs of X
    to minimize the entanglement across matrix with rows indexed by svd_legs. 

    Required Inputs:
    ---------------
    X        : NumPy array
    dis_legs : list of dimensions indicating legs the disentangler is applied to
    svd_legs : list of dimensions indicating legs for disentangling
    
    Optional Inputs:
    ---------------
    initial="identity" : initial disentangler, user can specify "random" or 2D NumPy array with compatible dimensions
    max_iterations=500 : maximum number of iterations of the selected optimizer
    min_grad_norm=1e-6 : termination threshold for norm of the gradient
    max_time=1e100     : maximum optimizer run time in seconds
    optimizer="CG"     : default "rCG"=Riemannian Conjugate Gradient
    objective=renyi    : objective function to optimize
    alpha=0.5          : parameter for renyi entropy
    chi=0              : parameter for trunc_error objective
    verbosity=0
    
    Temporary Inputs (for debugging):
    --------------------------------
    check_grad=False
    check_hess=False
    '''

    # ------------------ Check inputs ------------------ #
    # tensor dimensions
    assert all(0 <= d < X.ndim for d in dis_legs), "Invalid dimension in dis_legs"
    assert all(0 <= d < X.ndim for d in svd_legs), "Invalid dimension in svd_legs"
    dis_legs = sorted(set(dis_legs))
    svd_legs = sorted(set(svd_legs))

    n = np.prod([X.shape[d] for d in dis_legs]) # disentangler is n x n
    
    X_svd = ten_to_mat(X, svd_legs)
    s0 = np.linalg.svd(X_svd, compute_uv=False)

    # initial disentangler
    if initial == "identity":
        Q0 = np.eye(n)
    elif initial == "random":
        raise ValueError("initial option not supported")
    elif isinstance(Q, np.ndarray):
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Initial disentangler has incorrect dimensions.")
    else:
        raise TypeError("Initial disentangler must be 'identity', 'random', or a 2D NumPy array with compatible dimensions.")

    # objective parameters
    # Here we can check for possible mistakes when selecting combinations of objectives and parameters. 
    # For example, if the user specifies Renyi objective and also specifies truncation rank chi. 
    # Or if the user specifies trunc_error objective and also specifies an alpha...
    # Are there other combinations that the user should be warned of? 

    # optimizer parameters

    # ---------------- Alternating optimizer ---------------- #
    if optimizer == "alternating":
        Q = Q0
        cost = [np.linalg.norm(s0[chi:])]
        for i in range(max_iterations):
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
            
            if i>0 and np.abs(cost[-1]-cost[-2]) < min_grad_norm:
                if verbosity == 1:
                    print("exiting at iteration {0}".format(i))
                break

    # ---------------- Riemannian optimizer ---------------- #
    else:
        manifold = pymanopt.manifolds.Stiefel(n, n)

        @pymanopt.function.numpy(manifold)
        def cost(Q):
            return objective(Q, X, dis_legs, svd_legs, alpha, chi)[0]
        
        @pymanopt.function.numpy(manifold)
        def egrad(Q):
            return objective(Q, X, dis_legs, svd_legs, alpha, chi)[1]
        
        @pymanopt.function.numpy(manifold)
        def ehess(Q, E):
        # THIS FUNCTION DOES NOT WORK, YET!
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

            _, d_phi, dd_phi = objective(s[chi:])
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
                                   euclidean_hessian=ehess
                                   )
        if check_grad:
            diagnostics.check_gradient(problem)
        if check_hess:
            diagnostics.check_hessian(problem)
        
        if optimizer=="rCG":
            solver = pymanopt.optimizers.ConjugateGradient(max_iterations=max_iterations, 
                                                        max_time=max_time, 
                                                        min_gradient_norm=min_grad_norm, 
                                                        verbosity=verbosity
                                                        )
        elif optimizer=="rSD":
            solver = pymanopt.optimizers.SteepestDescent(max_iterations=max_iterations, 
                                                        max_time=max_time, 
                                                        min_gradient_norm=min_grad_norm, 
                                                        verbosity=verbosity
                                                        )
        else:
            raise ValueError("User specified optimizer is not supported")
        
        Q = solver.run(problem, initial_point=Q0).point

    U, S, V = disentangled_usv(X, Q, dis_legs, svd_legs)
    return Q, U, S, V