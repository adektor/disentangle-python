import numpy as np
import pymanopt
import time
import warnings

import pymanopt.tools
import pymanopt.tools.diagnostics

# TODO:
#       - cost, gradient, hessian have repeated operations...

# ------- Reshaping: tensor <-> matrix ------- #
def ten_to_mat(X, row_legs):
    ''' Reshapes a tensor X into a matrix X_mat with dimensions row_legs of X indexing the rows of X_mat
    Args
    ----
    X        : ND NumPy array
    row_legs : list of dimensions of X to use as rows in flattening

    Returns
    -------
    X_mat : 2D NumPy array (matrix) flattening of X
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
    ''' Reconstructs a tensor from its matrix form X_mat
    Args
    ----
    X_mat      : 2D NumPy array (matrix)
    orig_shape : original shape of the tensor before flattening
    row_legs   : list of dimensions that were used as rows in the matrix

    Returns
    -------
    X : ND NumPy array (tensor)
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
# -------------------------------------------- #

def disentangled_usv(X, Q, dis_legs, svd_legs):
    ''' Compute SVD across specified dimension after applying disentangler Q
    Args
    ----
    X        : NumPy array to be disentangled
    Q        : disentangler
    dis_legs : dimensions of X on which Q acts
    svd_legs : dimensions indicating which reshaping of X is SVD

    Returns
    -------
    u : left SVD factor of shape m x chi, where m = X[svd_legs[0]]*...*X[svd_legs[-1]]
    s : vector of chi singular values
    v : right SVD factor of shape chi x n, where n is the product of remaining dimensions of X
    '''
    
    QX_dis = Q @ ten_to_mat(X, dis_legs)
    QX = mat_to_ten(QX_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)
    return u, s, v

# -------------- Objective functions -------------- #
def nuclear(Q, X, dis_legs, svd_legs, alpha, chi):
    ''' Nuclear norm objective function (sum of singular values)
    Args
    ----
    Q        : disentangler
    X        : NumPy array to be disentangled
    dis_legs : dimensions of X on which Q acts
    svd_legs : dimensions indicating which reshaping of X is SVD
    alpha    : parameter (not used)
    chi      : parameter (not used)

    Returns
    -------
    cost  : objective function value
    egrad : Euclidean gradient of objective function wrt Q
    '''

    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)
    
    cost = np.sum(s)
    ds = np.ones_like(s)
    egrad = ten_to_mat(mat_to_ten(u@np.diag(ds)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T.conj())

    return cost, egrad


def truncate_svd(U, S, V, trunc_tol, chi):
    """
    Truncate SVD factors U, S, V with relative tolerance and max rank.

    Parameters
    ----------
    U, S, V : Factors from np.linalg.svd(A, full_matrices=False).
    trunc_tol : Relative truncation error tolerance.
    chi : Maximum allowed rank.

    Returns
    -------
    Utr, Str, Vtr : Truncated SVD factors.
    kept_rank : Number of singular values kept.
    rel_error : Relative truncation error achieved.
    """
    # squared singular values for Frobenius norm accounting
    svals_sq = S**2
    total_norm_sq = np.sum(svals_sq)

    # cumulative discarded norm
    disc_norm_sq = np.cumsum(svals_sq[::-1])[::-1]

    # relative error if truncating after k singular values
    rel_errs = np.sqrt(disc_norm_sq / total_norm_sq)

    # find smallest k s.t. error <= trunc_tol
    k_tol = np.searchsorted(rel_errs <= trunc_tol, True) + 1
    k_tol = min(len(S), k_tol)  # in case tol is too tight

    # final rank is min(chi, k_tol)
    k = min(chi, k_tol)

    Utr = U[:, :k]
    Str = S[:k]
    Vtr = V[:k, :]

    rel_error = np.sqrt(np.sum(svals_sq[k:]) / total_norm_sq)

    return Utr, Str, Vtr, rel_error


def renyi(Q, X, dis_legs, svd_legs, alpha, chi):
    ''' Renyi entropy objective function
    Args
    ----
    Q        : disentangler
    X        : NumPy array to be disentangled
    dis_legs : dimensions of X on which Q acts
    svd_legs : dimensions indicating which reshaping of X is SVD
    alpha    : parameter
    chi      : parameter (not used)

    Returns
    -------
    cost  : objective function value
    egrad : Euclidean gradient of objective function wrt Q
    '''

    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

    cost = 1/(1-alpha)*np.log(np.sum(s**(2*alpha)))

    fac = 2*alpha/(1-alpha)/np.sum(s**(2*alpha))
    ds = fac*s**(2*alpha - 1)
    egrad = ten_to_mat(mat_to_ten(u@np.diag(ds)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T.conj())
    
    return cost, egrad


def trunc_error(Q, X, dis_legs, svd_legs, alpha, chi):
    ''' Truncation error objective function (sum of trailing singular values squared)
    Args
    ----
    Q        : disentangler
    X        : NumPy array to be disentangled
    dis_legs : dimensions of X on which Q acts
    svd_legs : dimensions indicating which reshaping of X is SVD
    alpha    : parameter (not used)
    chi      : parameter - truncation rank

    Returns
    -------
    cost  : objective function value
    egrad : Euclidean gradient of objective function wrt Q
    '''

    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

    cost = np.sum(s[chi:]**2)

    ds = np.hstack([np.zeros(chi), 2*s[chi:]])
    ds_mat = u @ np.diag(ds) @ v

    dQX = ten_to_mat(mat_to_ten(ds_mat, X.shape, svd_legs), dis_legs)
    egrad = (dQX @ X_dis.conj().T)

    return cost, egrad


def trunc_error_hess(Q, E, X, dis_legs, svd_legs, alpha, chi):
    ''' Truncation error objective function (sum of trailing singular values squared)
    Args
    ----
    Q        : disentangler
    E        : matrix on which Hessian acts
    X        : NumPy array to be disentangled
    dis_legs : dimensions of X on which Q acts
    svd_legs : dimensions indicating which reshaping of X is SVD
    alpha    : parameter (not used)
    chi      : parameter - truncation rank

    Returns
    -------
    ehess  : Euclidean hessian at Q applied to E
    '''
    
    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)
    m, k, n = u.shape[0], u.shape[1], v.shape[1]

    ds = np.hstack([np.zeros(chi), 2*s[chi:]])
    egrad = ten_to_mat(mat_to_ten(u@np.diag(ds)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T.conj())

    EX_dis = E@ten_to_mat(X, dis_legs)
    EX_svd = ten_to_mat(mat_to_ten(EX_dis, X.shape, dis_legs), svd_legs)

    # F matrix (4.27)
    F = np.zeros([len(s), len(s)])
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                continue
            F[i, j] = 1/(s[j]**2 - s[i]**2)

    DUE = u @ (F*(u.T.conj() @ EX_svd @ v.T.conj() @ np.diag(s) + np.diag(s) @ v @ EX_svd.T.conj() @ u)) + \
          (np.eye(m) - u @ u.T.conj()) @ EX_svd @ v.T.conj() @ np.diag(1/s)

    d2fds2 = np.zeros(k)
    d2fds2[chi:] = 2

    Ds = np.diag(u.T.conj() @ EX_svd @ v.T.conj()) 
    DdfE = d2fds2*Ds

    DVE = v.T.conj() @ (F*(np.diag(s) @ u.T.conj() @ EX_svd @ v.T.conj() + v @ EX_svd.T.conj() @ u @ np.diag(s))) + \
          (np.eye(n) - v.T.conj() @ v) @ EX_svd.T.conj() @ u @ np.diag(1/s)
    
    ehess = ten_to_mat(mat_to_ten(DUE @ np.diag(ds) @ v + \
                                       u @ np.diag(DdfE) @ v + \
                                       u @ np.diag(ds) @ DVE.T.conj(), 
                                       X.shape, svd_legs), dis_legs)@ X_dis.T.conj()
    
    return ehess

def von_neumann(Q, X, dis_legs, svd_legs, alpha, chi):
    ''' Von-Neumann entropy objective function
    Args
    ----
    Q        : disentangler
    X        : NumPy array to be disentangled
    dis_legs : dimensions of X on which Q acts
    svd_legs : dimensions indicating which reshaping of X is SVD
    alpha    : parameter (not used)
    chi      : parameter (not used)

    Returns
    -------
    cost  : objective function value
    egrad : Euclidean gradient of objective function wrt Q
    '''

    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

    cost = -2*np.sum(s**2*np.log(s))
    ds = -2*s*(np.log(s**2) + 1)
    egrad = ten_to_mat(mat_to_ten(u@np.diag(ds)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T.conj())

    return cost, egrad
# ------------------------------------------------- #


def disentangle(X, dis_legs, svd_legs,
                initial="identity",
                max_iterations=1000,
                min_dQ=1e-6,
                min_grad_norm=1e-6,
                max_time=1e100,
                optimizer="rCG",
                objective=renyi,
                alpha=0.5,
                chi=0,
                man="Steifel",
                verbosity=0,
                return_log=False,
                check_grad=False,
                check_hess=False):
    '''
    Optimize a unitary matrix Q that contracts with dis_legs of X
    to minimize the entanglement across matrix with rows indexed by svd_legs. 

    Required Args
    -------------
    X        : NumPy array
    dis_legs : list of dimensions indicating legs the disentangler is applied to
    svd_legs : list of dimensions indicating legs for disentangling
    
    Kwargs
    ------
    initial="identity"  : initial disentangler, user can specify "random" or 2D NumPy array with compatible dimensions
    max_iterations=1000 : maximum number of iterations of the selected optimizer
    min_dQ=1e-6         : termination threshold for change in Q in alternating optimizer
    min_grad_norm=1e-6  : termination threshold for norm of the gradient
    max_time=1e100      : maximum optimizer run time in seconds
    optimizer="rCG"     : default "rCG"=Riemannian Conjugate Gradient
    objective=renyi     : objective function to optimize
    man="Steifel"       : manifold on which disentangler is optimized
    alpha=0.5           : parameter for renyi entropy
    chi=0               : parameter for trunc_error objective
    verbosity=0         : 1 print before and after optimization, 2 print every iteration of optimizer
    return_log=False    : return log of optimizer info
    
    Temporary Args (for debugging)
    ------------------------------
    check_grad=False
    check_hess=False

    Returns
    -------
    Q   : prod(X.shape[dis_dims]) x prod(X.shape[dis_dims])
    U   : prod(X.shape[svd_dims]) x chi
    S   : chi x chi
    V   : chi x prod(X.shape[svd_dims])

    log : if return_log is true
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
    elif isinstance(initial, np.ndarray):
        Q0 = initial
        if Q0.ndim != 2 or Q0.shape[0] != n or Q0.shape[1] != n:
            raise ValueError("Initial disentangler has incorrect dimensions.")
    else:
        raise TypeError("Initial disentangler must be 'identity', 'random', or a 2D NumPy array with compatible dimensions.")

    # possible mistakes in user-selected objectives and parameters
    if objective==renyi and chi != 0:
        warnings.warn("user-provided truncation rank 'chi' is not used in 'renyi' objective function", UserWarning)

    if objective==trunc_error:
        if alpha != 0.5:
            warnings.warn("user-provided parameter 'alpha' is not used in 'trunc_error' objective function", UserWarning)
        if chi<0:
            raise ValueError("user-provided truncation rank 'chi' must be positive")
        if not isinstance(chi, int):
            warnings.warn("user-provided truncation rank is a float... rounding to int", UserWarning)
            chi = round(chi)

    if objective==von_neumann:
        if chi !=0:
            warnings.warn("user-provided parameter 'chi' is not used in 'von_neumann' objective function", UserWarning)
        if alpha != 0.5:
            warnings.warn("user-provided parameter 'alpha' is not used in 'von_neumann' objective function", UserWarning)


    # ---------------- Alternating optimizer ---------------- #
    if optimizer.lower() in {"alternating", "alt"}:
        # check if the user specified incorrect or irrelevant parameters
        if min_grad_norm != 1e-6:
            warnings.warn("user-provided parameter 'min_grad_norm' is not used in alternating optimizer", UserWarning)
        if objective != trunc_error:
            warnings.warn("alternating optimizer only supports 'trunc_error' objective", UserWarning)
        if chi == 0:
            warnings.warn("For best results, set chi>0 in alternating optimizer", UserWarning)

        if verbosity>0:
            print("\nAlternating optimizer")
            print("Optimizing...")
        if verbosity>1:
            print("{:<10}  {:<25}  {:<15}".format("Iteration", "Cost", "Gradient norm"))
            print("{:<10}  {:<25}  {:<15}".format("--------", "-------------------------", "--------------"))

        start_time = time.time()

        Q = Q0
        cost = [np.linalg.norm(s0[chi:])]
        dQ = []
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

            M = ten_to_mat(QX_chi, dis_legs) @ (X_dis.conj().T)
            u, _, v = np.linalg.svd(M, full_matrices=False)
            Qnew = u@v
            dQ.append(np.linalg.norm(Qnew-Q))
            Q = Qnew

            if verbosity>1:
                print("{:<10d}  {:+.16e}  {:.8e}".format(i, cost[-1], dQ[-1]))

            # stopping conditions:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                if verbosity>0:
                    print("Terminated - Time limit reached in alternating optimizer. Exiting.")
                break
            if i>0 and dQ[-1]<min_dQ:
                if verbosity>0:
                    print("Terminated - min_dQ reached after {0} iterations, {1:.2f} seconds.".format(i, elapsed_time))
                break

        if verbosity>0 and i==max_iterations-1:
            print("Terminated - max iterations reached after {0} seconds".format(elapsed_time))
        
        if return_log:
            log = {"cost_history": cost,
                   "dQ_history"  : dQ, 
                   "iterations"  : i,
                   "runtime"     : elapsed_time}


    # ---------------- Riemannian optimizer ---------------- #
    else:
        # check if the user specified irrelevant parameters
        if min_dQ != 1e-6:
            warnings.warn("user-provided parameter 'dQ' is not used in Riemannian optimizers", UserWarning)

        if verbosity>0:
            print("\nRiemannian optimizer")
        if np.iscomplexobj(X) or man=='Unitary':
            manifold = pymanopt.manifolds.UnitaryGroup(n, retraction="polar")
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
            if objective==trunc_error:
                return trunc_error_hess(Q, E, X, dis_legs, svd_legs, alpha, chi)
            else:
                warnings.warn("user-selected cost function does not have Hessian support", UserWarning)
        
        problem = pymanopt.Problem(manifold=manifold, 
                                   cost=cost, 
                                   euclidean_gradient=egrad,
                                   euclidean_hessian=ehess
                                   )
        if check_grad:
            pymanopt.tools.diagnostics.check_gradient(problem)
        if check_hess:
            pymanopt.tools.diagnostics.check_hessian(problem)
        
        optimizer_args = dict(max_iterations=max_iterations,
                              max_time=max_time,
                              min_gradient_norm=min_grad_norm,
                              verbosity=verbosity,
                              log_verbosity=return_log
                              )
        
        if optimizer.lower() in {"rcg", "cg", "conjgrad", "conj_grad", "conjugate_gradient", "conjugategradient"}:
            solver = pymanopt.optimizers.ConjugateGradient(**optimizer_args)
        elif optimizer.lower() in {"rsd", "sd", "steepest_descent", "steepestdescent"}:
            solver = pymanopt.optimizers.SteepestDescent(**optimizer_args)
        elif optimizer.lower() in {"rtr", "tr", "trust_regions", "trustregions"}:
            solver = pymanopt.optimizers.TrustRegions(**optimizer_args)
        else:
            raise ValueError("User specified optimizer is not recognized")
        
        result = solver.run(problem, initial_point=Q0)
        # result = solver.run(problem)
        Q = result.point

        if return_log:
            log = {"cost_history"      : result.log["iterations"]["cost"],
                   "gradnorm_history"  : result.log["iterations"]["gradient_norm"],
                   "iterations"        : result.iterations,
                   "runtime"           : result.time}

    # ------------------ end optimizers ------------------ #

    # final disentangled SVD
    U, S, V = disentangled_usv(X, Q, dis_legs, svd_legs)

    if return_log:
        return Q, U, S, V, log
    else:
        return Q, U, S, V
    

def disentangle_bs(X, dis_legs, svd_legs, tol,
                max_dis=5,
                initial="identity",
                max_iterations=1000,
                min_dQ=1e-6,
                min_grad_norm=1e-6,
                max_time=1e100,
                man="Steifel",
                verbosity=0
                ):
    '''
    Disentangler binary search for determining a target rank chi and disentangler Q
    so that truncation error is less than tol. 

    Required Args
    -------------
    X        : NumPy array
    dis_legs : list of dimensions indicating legs the disentangler is applied to
    svd_legs : list of dimensions indicating legs for disentangling
    tol      : target truncation tolerance
    
    Kwargs
    ------
    max_dis=5           : maximum number of disentanglers computed (# of steps in binary search)
    initial="identity"  : initial disentangler, user can specify "random" or 2D NumPy array with compatible dimensions
    max_iterations=1000 : maximum number of iterations of the selected optimizer
    min_dQ=1e-6         : termination threshold for change in Q in alternating optimizer
    min_grad_norm=1e-6  : termination threshold for norm of the gradient
    max_time=1e100      : maximum optimizer run time in seconds
    man="Steifel"       : manifold on which disentangler is optimized
    verbosity=0         : 1 print before and after optimization, 2 print every iteration of optimizer
    
    Returns
    -------
    Q   : prod(X.shape[dis_dims]) x prod(X.shape[dis_dims])
    U   : prod(X.shape[svd_dims]) x chi
    S   : chi x chi
    V   : chi x prod(X.shape[svd_dims])

    log : if return_log is true
    '''

    # First disentangler is computed with Renyi-1/2 entropy
    Qbs, Ubs, Sbs, Vbs = disentangle(X, dis_legs, svd_legs, 
                        optimizer="rCG",
                        objective=renyi,
                        alpha=0.5,
                        initial=initial,
                        max_iterations=max_iterations,
                        min_grad_norm=min_grad_norm,
                        max_time=max_time,
                        man=man,
                        verbosity=verbosity-1
                        )
    
    # Based on truncation error pick left pointer, right pointer, and target rank
    kl = 0
    kr, rel_err = find_rank(Sbs, tol, chi=None)
    chi = kr
    iter = 1

    if verbosity >= 1:
        print(f"Initial Renyi rank = {chi}, truncation error = {rel_err:.2e}", flush=True)
    
    while kl <= kr and iter < max_dis:
        # get test rank k by bisection
        k = int(np.ceil((kl + kr)/2))

        if verbosity >= 1:
            print(f"[Iteration {iter}] testing rank k = {k} (kl={kl}, kr={kr})", flush=True)

        # Disentangle with test rank k
        Q, U, S, V = disentangle(X, dis_legs, svd_legs, 
                            optimizer="alternating", 
                            objective=trunc_error,
                            chi=k,
                            initial=initial,
                            min_dQ=min_dQ,
                            max_iterations=max_iterations,
                            max_time=max_time,
                            verbosity=verbosity-1
                            )
        # Relative truncation error
        rel_err = np.linalg.norm(S[k:]) / np.linalg.norm(S)
        
        if verbosity >= 1:
            print(f"              truncation error {rel_err:.2e}")
        
        # Update left pointer, right pointer, and valid truncation rank chi
        if rel_err <= tol:
            kr = k - 1
            chi = k
            Qbs, Ubs, Sbs, Vbs = Q, U, S, V
        else:
            kl = k + 1
        
        iter = iter + 1

    return Qbs, Ubs, Sbs, Vbs, chi


def find_rank(S, rel_tol, chi=None):
    """
    Find truncation rank k for singular values S
    based on relative truncation error and optional max rank chi.
    """
    svals_sq = S**2
    total_norm_sq = np.sum(svals_sq)

    # cumulative discarded norm if we truncate at k
    disc_norm_sq = np.cumsum(svals_sq[::-1])[::-1]  # shape (r,)
    rel_errs = np.sqrt(disc_norm_sq / total_norm_sq)

    # smallest k with rel_err <= tol
    k_tol = np.argmax(rel_errs <= rel_tol) + 1
    if k_tol == 0:  # in case no error is small enough
        k_tol = len(S)

    # apply max rank if provided
    if chi is not None:
        k = min(k_tol, chi)
    else:
        k = k_tol

    rel_error = np.sqrt(np.sum(svals_sq[k:]) / total_norm_sq)
    return k, rel_error