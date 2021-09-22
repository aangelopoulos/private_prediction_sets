import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.stats import binom, beta
from scipy.special import softmax
import pdb

def get_qtilde(n,alpha,gamma,epsilon,m):
    qtilde = (n+1)*(1-alpha)/(n*(1-gamma*alpha))+2/(epsilon*n)*np.log(m/(gamma*alpha))
    qtilde = min(qtilde, 1-1e-12)
    return qtilde

def generate_scores(n):
    return np.random.uniform(size=(n,))

def hist_2_cdf(cumsum, bins, n):
    def _cdf(t):
        if t > bins[-2]:
            return 1.0
        elif t < bins[1]:
            return 0.0
        else:
            return 1-cumsum[np.searchsorted(bins, t)]/n
    return _cdf

def get_private_quantile(scores, alpha, epsilon, gamma, bins):
    n = scores.shape[0]
    # Get the quantile
    qtilde = get_qtilde(n, alpha, gamma, epsilon, bins.shape[0])
    scores = scores.squeeze()
    score_to_bin = np.digitize(scores,bins)
    binned_scores = bins[np.minimum(score_to_bin,bins.shape[0]-1)]
    w1 = np.digitize(binned_scores, bins) 
    w2 = np.digitize(binned_scores, bins, right=True)
    # Clip bins
    w1 = np.maximum(np.minimum(w1,bins.shape[0]-1),0)
    w2 = np.maximum(np.minimum(w2,bins.shape[0]-1),0)
    lower_mass = np.bincount(w1,minlength=bins.shape[0]).cumsum()/qtilde
    upper_mass = (n-np.bincount(w2,minlength=bins.shape[0]).cumsum())/(1-qtilde)
    w = np.maximum( lower_mass , upper_mass ) 
    sampling_probabilities = softmax(-(epsilon/2)*w)
    # Check
    sampling_probabilities = sampling_probabilities/sampling_probabilities.sum()
    qhat = np.random.choice(bins,p=sampling_probabilities)
    return qhat

# Optimal gamma is a root.
def get_optimal_gamma(scores,n,alpha,m,epsilon):
    a = alpha**2
    b = - ( alpha*epsilon*(n+1)*(1-alpha)/2 + 2*alpha ) 
    c = 1
    best_q = 1
    gamma1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    gamma2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

    gamma1 = min(max(gamma1,1e-12),1-1e-12)
    gamma2 = min(max(gamma2,1e-12),1-1e-12)

    bins = np.linspace(0,1,m)

    q1 = get_private_quantile(scores, alpha, epsilon, gamma1, bins)
    q2 = get_private_quantile(scores, alpha, epsilon, gamma2, bins)

    return (gamma1, q1) if q1 < q2 else (gamma2, q2)

def get_optimal_gamma_m(n, alpha, epsilon):
    candidates_m = np.logspace(4,6,50).astype(int)
    scores = np.random.rand(n,1)
    best_m = int(1/alpha) 
    best_gamma = 1
    best_q = 1
    for m in candidates_m:
        gamma, q = get_optimal_gamma(scores,n,alpha,m,epsilon)
        if q < best_q:
            best_q = q
            best_m = m
            best_gamma = gamma
    return best_m, best_gamma

if __name__ == "__main__":
    n = 5000
    alpha = 0.1
    epsilon = 8 # removal definition, 5 is large.  usually we think of epsilon as 1 or 2.   
    mstar, gammastar = get_optimal_gamma_m(n, alpha, epsilon)
    scores = np.random.random((n,))
    q = get_private_quantile(scores, alpha, epsilon, gammastar, np.linspace(0,1,mstar))
    print(f"mstar: {mstar}, gammastar: {gammastar}, qhat: {q}")
