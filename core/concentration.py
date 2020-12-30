import numpy as np
import pickle as pkl
from tqdm import tqdm
import pdb

def cdf_of_sup_of_laplacian_process(m,eps):
    pass

def dkw(n,t):
    return 2*np.exp(-2*n*t*t)

def sup_of_laplacian_process(m,scale): # variance var, length of process m.
    L = np.random.laplace(loc=0.0, scale=scale, size=(m,))
    cumsum = np.cumsum(L)
    return np.max(np.abs(cumsum))

def get_cdf_of_process_supremum(num_replicates, m, scale):
    fname = f'.cache/{num_replicates}_{m}_{scale}_replicates.pkl'
    try:
        replicates = pkl.load( open(fname, 'rb') )
    except:
        replicates = [sup_of_laplacian_process(m, scale) for i in range(num_replicates)]
        replicates = np.sort(replicates)
        pkl.dump( replicates, open(fname, 'wb') )
    def _cdf(t):
        return np.searchsorted(replicates, t)/num_replicates
    return _cdf

if __name__ == "__main__":
    num_replicates = 1000000
    M = 1000 # max number of bins
    sens = 2
    epsilon = 5 
    scale = sens/epsilon 
    for m in tqdm(range(1,M+1)):
        cdf = get_cdf_of_process_supremum(num_replicates, m, scale)
