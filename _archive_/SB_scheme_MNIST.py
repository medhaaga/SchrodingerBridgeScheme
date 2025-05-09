import numpy as np
import os
from tqdm import tqdm
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--forward", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--step_size", type=float, default=100)
    parser.add_argument("--n_particles", type=int, default=1000)
    parser.add_argument("--sinkhorn_precision", type=float, default=1e-8)
    parser.add_argument("--sinkhorn_maxiter", type=int, default=1000000)

    return parser


def cost_matrix(X, Y):
    """L2 cost matrix
    """
    X, Y = torch.tensor(X), torch.tensor(Y)
    return ((torch.cdist(X, Y, p=2))**2).numpy()

def sinkhorn(cost_mat, a, b, epsilon, precision=1e-8, maxiter=1000):
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
    all_costs: list
        evolution of Schrodingerb cost with Sinkhorn iterations
    """
    a = a.reshape((cost_mat.shape[0], ))
    b = b.reshape((cost_mat.shape[1], ))
    K = np.exp(-cost_mat/epsilon)
    
    # initialization
    u = np.ones((cost_mat.shape[0], ))
    v = np.ones((cost_mat.shape[1], ))
    P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
    p_norm = np.trace(P.T @ P)
    all_costs = []

    for _ in range(maxiter):
        u = a/np.maximum((K @ v), 1e-300) # avoid divided by zero
        v = b/np.maximum((K.T @ u), 1e-300)
        P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
        if abs((np.trace(P.T @ P) - p_norm)/p_norm) < precision:
            break
        p_norm = np.trace(P.T @ P)
        cost = np.trace(cost_mat.T @ P)
        all_costs.append(cost)
    return cost, P, all_costs
    
def entropy_SB_scheme_sinkhorn(X, steps=[1], eps=0.01, precision=1e-8, maxiter=1e6, forward=True, dir=None):

    """
    Implements the Sinkhorn-based approximation of explicit Euler discretization (forward & reverse) of the gradient flow of entropy function.

    Arguments
    -----------------------
    X: np array-like 
        All particles sampled from the starting distribution.

    steps: list, optional, default=[1]
        List of time steps at which to record the state of the array X.

    eps: float, optional, default=0.01
        Epsilon value for the Sinkhorn algorithm.

    precision: float, optional, default=1e-8
        Precision for the Sinkhorn algorithm convergence.

    maxiter: int, optional, default=1e6
        Maximum number of iterations for the Sinkhorn algorithm.

    forward: bool, optional, default=True
        Direction of the Euler discretization. If True, perform forward discretization, otherwise reverse.

    dir: str or None, optional, default=None
        Directory path to save the state of X at specified steps. If None, states are not saved.

    Returns
    -----------------------
    np.array
        Array containing the state of X at the specified steps.
    """

    start = X
    n, d = X.shape[0], X.shape[1]
    total_steps = steps[-1]
    X_list = [X]
    if dir:
        np.save(os.path.join(dir, f'eps{args.step_size}.npy'), np.array(X_list))
        

    for i in tqdm(range(total_steps)):
   
        cost_mat = cost_matrix(X, X)

        _, avg_plan, _ = sinkhorn(cost_mat, np.ones(n)/n, np.ones(n)/n, epsilon=eps, precision=precision, maxiter=maxiter)
        bar_proj = n*np.matmul(avg_plan, X.reshape(n,d)).reshape(n,d)
        if forward:
            X = 2*X - bar_proj
        else:
            X = bar_proj

        if i+1 in steps:
            X_list.append(X)
            print(f'Epsilon: {eps}, Steps: {i+1}, Distance from start: {np.linalg.norm(start-X)}')
            if dir:
                np.save(os.path.join(dir, f'eps{args.step_size}.npy'), np.array(X_list))

    return np.array(X_list)


if __name__ == '__main__':  

    parser = parse_arguments()
    args = parser.parse_args()
    
    ##########################################
    ## Create source data
    ##########################################

    # Define transformations for the dataset (e.g., normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Load both train and test portions of MNIST
    train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='data', train=False, transform=transform, download=True)

    # Concatenate the train and test datasets into one dataset
    full_dataset = ConcatDataset([train_dataset, test_dataset])

    dataloader = DataLoader(full_dataset, shuffle=True, batch_size=args.n_particles)

    X, _ = next(iter(dataloader))
    steps = 100*np.arange(1, (args.max_steps//100)+1)

    dir = f'results/FashionMNIST'
    os.makedirs(dir, exist_ok=True)

    X_SB_estimate = entropy_SB_scheme_sinkhorn(X.numpy(), steps=steps, eps=args.step_size, precision=args.sinkhorn_precision, maxiter=args.sinkhorn_maxiter, dir=dir)

    np.save(os.path.join(dir, f'eps{args.step_size}.npy'), X_SB_estimate)
    print(f'Save {X_SB_estimate.shape} sized SB scheme steps.')

