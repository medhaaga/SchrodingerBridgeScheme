import numpy as np
import os
import argparse
from tqdm import tqdm
from estimate_SB import schbridge, sinkhorn, cost_matrix


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--SB_estimation_method", type=str, choices=['mcmc', 'sinkhorn'], default='sinkhorn')
    parser.add_argument("--forward", type=int, choices=[0, 1], default=1)
    parser.add_argument("--source_dist", type=str, choices=['gaussian', 'thin_gaussian', 'gaussian_mix'], default='gaussian')
    parser.add_argument("--time", type=float, default=2.)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument("--mcmc_steps", type=int, default=1e6)
    parser.add_argument("--mcmc_burn", type=int, default=1e4)

    return parser
 
def entropy_SB_scheme_mcmc(X, steps=[1], eps=0.001, total=1e6, discard=1e4, forward=True):
    start = X
    n = X.shape[0]
    total_steps = steps[-1]
    X_list = []
    for i in tqdm(range(total_steps)):
        cost_mat = cost_matrix(X, X)
        _, avg_plan, _, _ = schbridge(cost_mat, eps=eps, total=total, discard=discard)
        bar_proj = n*np.matmul(avg_plan, X.reshape(n,1)).reshape(n,)

        if forward:
            X = 2*X - bar_proj
        else:
            X = bar_proj

        if i+1 in steps:
            X_list.append(X)
            print(f'Epsilon: {eps}, Steps: {i+1}, Distance from start: {np.linalg.norm(start-X_list[-1])}')

    return np.array(X_list)

def entropy_SB_scheme_sinkhorn(X, steps=[1], eps=0.001, precision=1e-8, maxiter=10000, forward=True):
    start = X
    n = X.shape[0]
    total_steps = steps[-1]
    X_list = []
    for i in tqdm(range(total_steps)):
   
        cost_mat = cost_matrix(X, X)
        _, avg_plan = sinkhorn(cost_mat, np.ones(n)/n, np.ones(n)/n, epsilon=eps, precision=precision, maxiter=maxiter)
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

    parser = parse_arguments()
    args = parser.parse_args()

    ##########################################
    ## Create source data
    ##########################################
    

    n = args.n_particles # number of particles
    if args.source_dist == 'gaussian':
        mu, sigma_squared, sigma = 0, 4, 2
        X = np.random.normal(mu, sigma, n)
    elif args.source_dist == 'thin_gaussian':
        mu, sigma_squared, sigma = 0, .25, .5
        X = np.random.normal(mu, sigma, n)
    elif args.source_dist == 'gaussian_mix':
        mu1, mu2 = -2, 2
        X = np.concatenate((np.random.normal(-mu1, 1, 500//2), np.random.normal(mu2, 1, 500//2)))


    # Calculate cost matrix
    cost_mat = cost_matrix(X, X)

    direction = "forward" if args.forward else "reverse"
    max_steps = int(args.time/args.step_size)
    steps = 10*np.arange(1, (max_steps//10)+1)
    print(steps)

    if args.SB_estimation_method == 'mcmc':
        X_SB_estimate = entropy_SB_scheme_mcmc(X, steps=steps, eps=args.step_size, total=args.mcmc_steps, discard=args.mcmc_burn, forward=args.forward)
    else:
        X_SB_estimate = entropy_SB_scheme_sinkhorn(X, steps=steps, eps=args.step_size, forward=args.forward)

    # Save arrays 
    dir = f'{args.SB_estimation_method}/{direction}'
    os.makedirs(dir, exist_ok=True)

    np.save(os.path.join(dir, f'eps{args.step_size}.npy'), X_SB_estimate)
    print(f'Save {X_SB_estimate.shape} sized SB scheme steps.')

    print(f'Epsilon: {args.step_size} -->  Distance from start: {np.linalg.norm(X - X_SB_estimate[-1])}')
    
