import numpy as np

# ------- Reshaping: tensor <-> matrix ------- #
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
# -------------------------------------------- #


# -------------- Objective functions -------------- #
def nuclear(Q, X, dis_legs, svd_legs, alpha, chi):
    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)
    
    cost = np.sum(s)
    ds = np.ones_like(s)
    egrad = ten_to_mat(mat_to_ten(u@np.diag(ds)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T)

    return cost, egrad

def renyi(Q, X, dis_legs, svd_legs, alpha, chi):
    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

    cost = 1/(1-alpha)*np.log(np.sum(s**(2*alpha)))

    fac = 2*alpha/(1-alpha)/np.sum(s**(2*alpha))
    ds = fac*s**(2*alpha - 1)
    egrad = ten_to_mat(mat_to_ten(u@np.diag(ds)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T)
    
    return cost, egrad

def trunc_error(Q, X, dis_legs, svd_legs, alpha, chi):
    X_dis = ten_to_mat(X, dis_legs)
    QX = mat_to_ten(Q@X_dis, X.shape, dis_legs)
    QX_svd = ten_to_mat(QX, svd_legs)
    u, s, v = np.linalg.svd(QX_svd, full_matrices=False)

    cost = np.sum(s[chi:]**2)
    ds = np.hstack([np.zeros(chi), 2*s[chi:]])
    egrad = ten_to_mat(mat_to_ten(u@np.diag(ds)@v, X.shape, svd_legs), dis_legs) @ (X_dis.T)

    return cost, egrad