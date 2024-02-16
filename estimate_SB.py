import numpy as np
from numba import jit, njit, prange, int64, float64, boolean
from tqdm import tqdm

@jit(nopython=True)
def _gibbs_one_iter(cost_mat, eps, perm, cost):
    """One interation in Gibbs sampling.

    Randomly sample a pair (i, j) and test for swapping in current permutation
    ``perm``.

    Parameters
    ----------
    cost_mat : array-like, shape (size, size)
        Cost matrix between two samples.
    eps : float
        Regularization parameter.
    perm : array-like
        Current permutation.
    cost : float
        Current cost.

    Returns
    -------
    cost : float
        The cost after this iteration.
    swap : bool
        Indication of swapping.
    perm : array-like, shape (size,) 
        Accepeted permutation from the single MH step.
    """
    n = cost_mat.shape[0]
    i, j = np.random.choice(n, size=2, replace=False)

    # swap based on the conditional probability
    diff_cost = cost_mat[i, perm[i]] + cost_mat[j, perm[j]] - \
        cost_mat[i, perm[j]] - cost_mat[j, perm[i]]
    prob = np.exp(diff_cost/eps)
    swap = 0

    u = np.random.uniform(low=0, high=1, size=1)

    if u <= prob:
        swap = 1
        perm[i], perm[j] = perm[j], perm[i]
        cost -= diff_cost
        
    return cost, swap, perm

@jit(nopython=True)
def update_indices(arr, x_indices, y_indices, value):
    # Assume arr is a 2D array
    # arr[x_indices, y_indices] += value
    n = len(x_indices)
    
    for i in prange(n):
        x = x_indices[i]
        y = y_indices[i]
        arr[x, y] += value
    
    return arr

@jit(nopython=True)
def schbridge(cost_mat, eps, total, discard):
    """Compute EOT statistic with Gibbs sampling.

    Parameters
    ----------
    cost_mat : array-like, shape (size, size)
        Cost matrix between two samples.
    eps : float
        Regularization parameter.
    total : int
        Maximum number of Gibbs samples generated to compute the statistic.
    discard : int
        Number of Gibbs samples discarded.
    log : bool, optional
        Indication of storing the cost of each Gibbs sample, by default ``False``.

    Returns
    -------
    avg_cost : float
        Estinmated Schrodinger cost via MCMC samples 
    avg_plan : array-like, shape (size, size)
        Estinmated Schrodinger bridge via MCMC samples 
    """
    n = cost_mat.shape[0]

    # initial permutation: the identity map
    perm = np.arange(n)
    # initial cost
    cost = np.sum(np.diag(cost_mat))

    # store results
    costs = []
    avg_plan = np.zeros((n, n))
    avg_cost = 0.0

    # counters
    cnt = 0
    swaps = 0
    
    for t in range(total):
        cost, swap, perm = _gibbs_one_iter(cost_mat, eps, perm, cost)

        # store results
        if t >= discard:
            swaps += swap 
            avg_plan *= cnt/(cnt + 1)
            avg_plan = update_indices(avg_plan, np.arange(n), perm, 1/n/(cnt + 1))
            # avg_plan[np.arange(n), perm] += 1/n/(cnt + 1)
            avg_cost = avg_cost*cnt/(cnt + 1) + cost/n/(cnt + 1)
            cnt += 1
            costs.append(cost/n)

    return avg_cost, avg_plan, np.array(costs), swaps/(total - discard)


element_max = np.vectorize(max)
def sinkhorn(cost_mat, a, b, epsilon, precision, maxiter=1000):
    """Computes EOT statistic with Sinkhorn algorithm.

    Parameters
    ----------
    cost_mat : array-like, shape (size, size)
        Cost matrix between two samples.
    a : array-like, shape (size,)
        Discrete probability distribution along first marginal.  
    b : array-like, shape (size,)
        Discrete probability distribution along second marginal.  
    epsilon : float
        Regularization parameter.
    precision : float
        Precision for when to stop the Sinkhorn update.

    Returns
    -------
    cost : float
        Computed EOT cost
    P : array-like, shape (size, size)
        Minimizer of EOT
    """
    a = a.reshape((cost_mat.shape[0], 1))
    b = b.reshape((cost_mat.shape[1], 1))
    K = np.exp(-cost_mat/epsilon)
    
    # initialization
    u = np.ones((cost_mat.shape[0], 1))
    v = np.ones((cost_mat.shape[1], 1))
    P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
    p_norm = np.trace(P.T @ P)
    
    for _ in range(maxiter):
        u = a/element_max((K @ v), 1e-300) # avoid divided by zero
        v = b/element_max((K.T @ u), 1e-300)
        P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
        if abs((np.trace(P.T @ P) - p_norm)/p_norm) < precision:
            break
        p_norm = np.trace(P.T @ P)
    cost = np.trace(cost_mat.T @ P)
    return cost, P

def cost_matrix(X, Y):
    """L2 cost matrix
    """
    n = X.shape[0]
    return (X.reshape((n,1)) - X.reshape((1,n)))**2

def entropy_SB_scheme_gibbs(X, steps=[1], eps=0.001, total=1e6, discard=1e4, forward=True):
    start = X
    n = X.shape[0]
    total_steps = steps[-1]
    X_list = []
    for i in tqdm(range(total_steps)):
        cost_mat = cost_matrix(X, X)
        avg_cost, avg_plan, _, _ = schbridge(cost_mat, eps=eps, total=total, discard=discard)
        bar_proj = n*np.matmul(avg_plan, X.reshape(n,1)).reshape(n,)

        if forward:
            X = 2*X - bar_proj
        else:
            X = bar_proj

        if i+1 in steps:
            X_list.append(X)
            print(f'Epsilon: {eps}, Steps: {i+1}, Distance from start: {np.linalg.norm(start-X_list[-1])}')

    return np.array(X_list)

def entropy_SB_scheme_sinkhorn(X, steps=[1], eps=0.001, precision=1e-30, maxiter=10000, forward=True):
    start = X
    n = X.shape[0]
    total_steps = steps[-1]
    X_list = []
    for i in tqdm(range(total_steps)):
        cost_mat = cost_matrix(X, X)
        avg_cost, avg_plan = sinkhorn(cost_mat, np.ones(n)/n, np.ones(n)/n, epsilon=eps, precision=precision, maxiter=maxiter)
        bar_proj = n*np.matmul(avg_plan, X.reshape(n,1)).reshape(n,)
        if forward:
            X = 2*X - bar_proj
        else:
            X = bar_proj

        if i+1 in steps:
            X_list.append(X)
            print(f'Epsilon: {eps}, Steps: {i+1}, Distance from start: {np.linalg.norm(start-X)}')
    return np.array(X_list)



if __name__ == '__main__':
    n = 100
    eps = 0.01
    mu, sigma_squared, sigma = 0, 4, 2
    X = np.random.normal(mu, sigma, n)
    cost_mat = cost_matrix(X, X)
    gibbs_cost, gibbs_plan, costs, accept = schbridge(cost_mat, eps=eps, total=1000, discard=500)