import numpy as np
from scipy.special import expit
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from estimate_SB import _gibbs_one_iter, schbridge, sinkhorn, cost_matrix

def custom_resample(original_array, acceptance_probs):
    n = len(original_array)
    resampled_indices = []
    first_occurrence_indices = {}

    while len(set(resampled_indices)) < n:
        # Use numpy's random.choice to resample indices with specified probabilities
        chosen_index = np.random.choice(n, p=acceptance_probs)
        # print(chosen_index)

        # Mark the index as resampled
        resampled_indices.append(chosen_index)

        # Record the first occurrence index for each unique element
        element = original_array[chosen_index]
        if element not in first_occurrence_indices:
            first_occurrence_indices[element] = len(resampled_indices)-1

    # Create the resampled array
    resampled_array = original_array[list(resampled_indices)]

    # Create a list of first occurrence indices for each unique element
    first_occurrence_indices_list = [first_occurrence_indices[element] for element in original_array]

    return resampled_array, first_occurrence_indices_list

def kl_SB_scheme_gibbs(X, steps=[1], eps=0.001, total=1e6, discard=1e4, forward=True):
    start = X
    n = X.shape[0]
    total_steps = steps[-1]
    X_list = []

    for i in tqdm(range(total_steps)):

        surrogate_samp, first_occurrence_indices_list = custom_resample(X, np.exp(-(X**2)/2)/sum(np.exp(-(X**2)/2)))
        m = surrogate_samp.shape[0]

        cost_mat = cost_matrix(surrogate_samp, surrogate_samp)
        avg_cost, avg_plan, _, _ = schbridge(cost_mat, eps=eps, total=total, discard=discard)

        weighted_avg = np.matmul(avg_plan, surrogate_samp.reshape(m,1)).reshape(m,)

        bar_proj = m*weighted_avg[list(first_occurrence_indices_list)]

        if forward:
            X = bar_proj
        else:
            X = 2*X - bar_proj

        if i+1 in steps:
            X_list.append(X)
            print(f'Epsilon: {eps}, Steps: {i+1}, Distance from start: {np.linalg.norm(start-X_list[-1])}')

    return np.array(X_list)




if __name__ == '__main__':  

    ##########################################
    ## Forward process
    ##########################################
    

    n = 500 # number of particles
    mu, sigma_squared, sigma = 0, 4, 2
    X = np.random.normal(mu, sigma, n)
    cost_mat = cost_matrix(X, X)

    for e in [0.1, 0.01, 0.001]: 

        max_steps = int(2/e)
        steps = 10*np.arange(1, (max_steps//10)+1)

        ## Gibbs
        X_SB_Gibbs = kl_SB_scheme_gibbs(X, steps=steps, eps=e, total=1e6, discard=1e4, forward=True)

        # Save multiple arrays into a single .npz file
        file_path = f'X_SB_Gibbs_forward_eps{e}.npy'
        np.save(file_path, X_SB_Gibbs)


        print(f'Epsilon: {e} -->  Gibbs: {np.linalg.norm(X - X_SB_Gibbs[-1])}')

    ##########################################
    ## Reverse process
    ##########################################
    

    n = 500 # number of particles
    mu, sigma_squared, sigma = 0, 6, np.sqrt(6)
    X_ = np.random.normal(mu, sigma, n)
    cost_mat = cost_matrix(X_, X_)

    for e in [0.1, 0.01, 0.001]: 

        max_steps = int(2/e)
        steps = 10*np.arange(1, (max_steps//10)+1)

        ## Gibbs
        X_SB_Gibbs = kl_SB_scheme_gibbs(X_, steps=steps, eps=e, total=1e6, discard=1e4, forward=False)

        # Save multiple arrays into a single .npz file
        file_path = f'X_SB_Gibbs_reverse_eps{e}.npy'
        np.save(file_path, X_SB_Gibbs)


        print(f'Epsilon: {e} -->  Gibbs: {np.linalg.norm(X_ - X_SB_Gibbs[-1])}')

