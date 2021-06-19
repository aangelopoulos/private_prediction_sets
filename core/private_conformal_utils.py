import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
from core.concentration import dkw, get_cdf_of_process_supremum, pointwise_cdf_bound
from scipy.optimize import brentq
from scipy.stats import binom, beta
from scipy.special import softmax
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
    cumsum = np.cumsum(hist[::-1])[::-1]
    #cumsum = cumsum.sum() - cumsum
    return hist, cumsum

def hist_2_cdf(cumsum, bins, n):
    def _cdf(t):
        if t > bins[-2]:
            return 1.0
        elif t < bins[1]:
            return 0.0
        else:
            return 1-cumsum[np.searchsorted(bins, t)]/n
    return _cdf

def get_adjusted_alpha_cdf(n, alpha, gamma):
    return 1-(n+1)*(1-alpha)/(n*(1-gamma*alpha))

def get_optimal_gamma(n,alpha,m,epsilon,num_replicates):
    gammas = np.linspace(0,0.2,1000)
    best_gamma = 1
    best_value = 1
    scale = 2/epsilon
    sup_lproc_cdf = get_cdf_of_process_supremum(num_replicates,m,scale)
    for gamma in gammas:
        def _laplace_condition(q):
            return sup_lproc_cdf(q) - (1-gamma*alpha)
        value = (n+1)*(1-alpha)/(n*(1-gamma*alpha)) + brentq(_laplace_condition,0,n)/n
        if value < best_value:
            best_gamma = gamma
            best_value = value
    return best_gamma, best_value

def get_qhat(n,alpha,epsilon,gamma,bins,num_replicates):
    m = bins.shape[0] - 1
    scale = 2/epsilon
    sup_lproc_cdf = get_cdf_of_process_supremum(num_replicates,m,scale)
    def _laplace_condition(q):
        return sup_lproc_cdf(q) - (1-gamma*alpha)
    laplace_quantile = brentq(_laplace_condition,0,n)
    adjusted_quantile = 1-get_adjusted_alpha_cdf(n, alpha, gamma) + laplace_quantile/n
    return adjusted_quantile

def get_private_quantile(scores, alpha, epsilon, gamma, bins, num_replicates):
    n = scores.shape[0]
    scores = scores.squeeze()
    score_to_bin = np.digitize(scores,bins)
    binned_scores = bins[np.minimum(score_to_bin,bins.shape[0]-1)]
    sort_idx = binned_scores.argsort().argsort()
    ranks = (np.array(range(n))+1.0)[sort_idx]
    alpha_adjusted = (n+1)*(1-alpha)/(n*(1-gamma*alpha)) + 1/epsilon/n * np.log(bins.shape[0]/gamma/alpha) + 1/n
    sampling_probabilities = softmax(-(epsilon/2)*np.abs(alpha_adjusted * n - ranks))
    return np.random.choice(binned_scores,p=sampling_probabilities)

#def get_private_quantile(scores, alpha, epsilon, gamma, bins, num_replicates):
#    n = scores.shape[0]
#    hist, cumsum = private_hist(scores, epsilon, bins)
#    ecdf = hist_2_cdf(cumsum, bins,n)
#    qhat = get_qhat(n,alpha,epsilon,gamma,bins,num_replicates)
#    if qhat > 1-1e-5:
#        return bins[-1]
#    def _condition(q):
#        return ecdf(q) - qhat 
#    shat = brentq(_condition, 1e-5, 1-1e-5)
#    bin_idx = min(np.argmax(bins > shat)+1,bins.shape[0]-1) # handle rounding up
#    return bins[bin_idx] 

def get_mstar(n, alpha, epsilon, gamma, num_replicates):
    candidates = np.arange(10,int(0.1*n),50)
    scores = np.random.rand(n,1)
    best_m = 10
    best_q = 1
    print(r'Calculate $M^*$')
    for m in tqdm(candidates):
        q = get_private_quantile(scores, alpha, epsilon, gamma, np.linspace(0,1,m), num_replicates)
        if q < best_q:
            best_q = q
            best_m = m
    print(best_m)
    print(best_q)
    return best_m

if __name__ == "__main__":
    M = 100 # max number of bins
    #m = 10
    num_replicates=100000
    n = 1000
    alpha = 0.1
    epsilon = 2 # removal definition, 5 is large.  usually we think of epsilon as 1 or 2.   
    bins = np.linspace(0,1,M)
    #plot_beta_inv(n, m, scale)
    scores = generate_scores(n)
    gamma, _ = get_optimal_gamma(n,alpha,M,num_replicates)
    qhat = get_private_quantile(scores, alpha,  epsilon, gamma, bins, num_replicates)
    print(qhat)
    mstar = get_mstar(n, alpha, epsilon, gamma, num_replicates)
    print(mstar)
    # we would like qhat to be larger than 1-alpha
    print("hi")
