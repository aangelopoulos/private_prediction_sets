import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
from concentration import dkw, get_cdf_of_process_supremum#(num_replicates, m, scale)
from scipy.optimize import brentq
import pdb

def beta(t,n,m,scale,num_replicates):
    sup_lproc_cdf = get_cdf_of_process_supremum(num_replicates,m,scale)
    return dkw(n,.85*t) + 1 - sup_lproc_cdf(.15*n*t) # some magic numbers

def beta_inv(alpha, n, m, scale, num_replicates):
    def _condition(t):
        return beta(t, n, m, scale, num_replicates)-alpha
    return brentq(_condition,1e-9,1-1e-9)

def plot_beta(n, m, scale, num_replicates=100000):
    ts = np.linspace(0,1,100)
    bs = np.array([beta(t,n,m,scale,num_replicates) for t in ts])
    plt.plot(ts,bs)
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel(r'$\beta$(t)')
    plt.savefig('./beta_plot.pdf')

def plot_beta_inv(n, m, scale, num_replicates=100000):
    alphas = np.linspace(1e-4,0.2,100)
    binvs = np.array([beta_inv(alpha,n,m,scale,num_replicates) for alpha in alphas])
    plt.plot(alphas,binvs, linewidth=3)
    plt.plot(alphas,alphas,color='#aaaaaa',linestyle='dashed')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta^{-1}(\alpha)$')
    plt.savefig('./beta_inv_plot.pdf')

def generate_scores(n):
    return np.random.uniform(size=(n,))

def private_hist(scores,epsilon,bins):
    scale = 2/epsilon
    hist, _ = np.histogram(scores, bins=bins)
    hist = hist + np.random.laplace(loc=0.0,scale=scale,size=hist.shape)
    cumsum = np.cumsum(hist)
    return hist, cumsum

def hist_2_cdf(cumsum, bins):
    def _cdf(t):
        return cumsum[np.searchsorted(bins[:-2], t)]/cumsum[-1]
    return _cdf

def get_private_quantile(scores, alpha, epsilon, bins, num_replicates):
    hist, cumsum = private_hist(scores, epsilon, bins)
    ecdf = hist_2_cdf(cumsum, bins)
    def _condition(q):
        return ecdf(q) - (1 - alpha + beta_inv(alpha, scores.shape[0], np.searchsorted(bins[:-1],q), 2/epsilon, num_replicates=num_replicates) )
    return brentq(_condition, 1e-5, 1-1e-5)
    

if __name__ == "__main__":
    M = 1000 # max number of bins
    #m = 10
    num_replicates=100000
    n = 10000
    alpha = 0.05
    epsilon = 5
    bins = np.linspace(0,1,M)
    #plot_beta_inv(n, m, scale)
    scores = generate_scores(n)
    qhat = get_private_quantile(scores, alpha,  epsilon, bins, num_replicates)
    print(qhat)
    # we would like qhat to be larger than 1-alpha
    print("hi")
