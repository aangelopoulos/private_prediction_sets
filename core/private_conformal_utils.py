import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
from core.concentration import dkw, get_cdf_of_process_supremum, pointwise_cdf_bound
from scipy.optimize import brentq
from scipy.stats import binom, beta
import pdb

def beta_dkw(t,n,m,scale,num_replicates):
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
        if t > bins[-1]:
            return 1.0
        elif t < bins[1]:
            return 0.0
        else:
            return cumsum[np.searchsorted(bins, t)-1]/cumsum[-1]
    return _cdf

def get_adjusted_alpha_cdf(n, alpha, g1, g2):
    def _condition(mprime):
        return beta.cdf(1-alpha*g1,mprime,n-mprime+1) - g2*alpha
    return 1-brentq(_condition, 1, n)/n

def get_private_quantile(scores, alpha, epsilon, gammas, bins, num_replicates):
    hist, cumsum = private_hist(scores, epsilon, bins)
    ecdf = hist_2_cdf(cumsum, bins)
    n = scores.shape[0]
    m = bins.shape[0] - 1
    scale = 2/epsilon
    g1, g2 = gammas
    g3 = 1-g1-g2
    sup_lproc_cdf = get_cdf_of_process_supremum(num_replicates,m,scale)
    def _laplace_condition(q):
        return sup_lproc_cdf(q) - (1-g3*alpha)
    laplace_quantile = brentq(_laplace_condition,0,n)
    adjusted_quantile = 1-get_adjusted_alpha_cdf(n, alpha, g1, g2) + laplace_quantile/n
    if adjusted_quantile > 1-1e-5:
        return bins[-1]
    def _condition(q):
        return ecdf(q) - adjusted_quantile
    qhat = brentq(_condition, 1e-5, 1-1e-5)
    bin_idx = min(np.argmax(bins > qhat)+1,bins.shape[0]-1) # handle rounding up
    return bins[bin_idx] 

if __name__ == "__main__":
    M = 10 # max number of bins
    #m = 10
    num_replicates=100000
    n = 1000
    alpha = 0.1
    epsilon = 1 # removal definition, 5 is large.  usually we think of epsilon as 1 or 2.   
    bins = np.linspace(0,1,M)
    #plot_beta_inv(n, m, scale)
    scores = generate_scores(n)
    gammas = (0.9,1/50)
    qhat = get_private_quantile(scores, alpha,  epsilon, gammas, bins, num_replicates)
    print(qhat)
    pdb.set_trace()
    # we would like qhat to be larger than 1-alpha
    print("hi")
