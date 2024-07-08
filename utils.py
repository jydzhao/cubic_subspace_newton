import numpy as np

def save_run(filename, A, b, lam, beta, results):
    with open(filename, 'wb') as f:
        np.save(f, A)
        np.save(f, b)
        np.save(f, lam)
        # np.save(f, mu)
        np.save(f, beta)
        np.save(f, np.array(results, dtype=object), allow_pickle=True)

def load_run(filename):
    # to load data again:
    with open(filename, 'rb') as f:
        A = np.load(f)
        b = np.load(f)
        lam = np.load(f)
        # mu = np.load(f)
        beta = np.load(f)
        results = np.load(f, allow_pickle=True)      
    
    return A, b, lam, beta, results

def generate_cov_mat(n, num_rows_to_be_filled, eps_range, seed=31415):

    np.random.seed(seed)
    
    cov_mat = np.eye(n)
    
    assert(eps_range >= 0), 'eps_range has to be non-negative'
    assert(eps_range <= 1), 'eps_range has to be smaller than 1'

    for row in range(num_rows_to_be_filled):
        eps = np.random.uniform(-eps_range,+eps_range)
    
        for k in range(n-row-1):
            cov_mat[k,k+row+1] = eps
            cov_mat[k+row+1,k] = eps
    
    return 1/np.sqrt(n) * cov_mat @ cov_mat.T

def corr_value(X):
    C = np.corrcoef(X)
    return 1/(X.shape[0]) * (np.linalg.norm(C, ord='fro') - np.sqrt(X.shape[0])) # subtract diagonal entries, which are 1, 1, ..., 1