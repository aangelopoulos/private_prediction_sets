import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from core.private_conformal_utils import *
import pdb

def get_qhats_ns(ns, alpha, epsilons_small):
    qhats_ns = np.zeros((len(epsilons_small),ns.shape[0]))
    for i in range(len(epsilons_small)):
        for j in range(ns.shape[0]):
            try:
                n = ns[j]
                epsilon = epsilons_small[i]
                mstar, gammastar = get_optimal_gamma_m(n, alpha, epsilon)
                qhats_ns[i,j] = get_qtilde(n,alpha,gammastar,epsilon,mstar)
                #print(f"n:{n}, epsilon:{epsilon}, mstar:{mstar}, gammastar:{gammastar}, qhat:{qhats_ns[i,j]:.3f}")
            except:
                qhats_ns[i,j] = None
    return qhats_ns

def get_qhats_epsilons(epsilons, alpha, ns_small):
    qhats_epsilons = np.zeros((len(ns_small),epsilons.shape[0]))
    for i in range(len(ns_small)):
        for j in range(epsilons.shape[0]):
            try:
                n = ns_small[i]
                epsilon = epsilons[j]
                mstar, gammastar = get_optimal_gamma_m(n, alpha, epsilon)
                qhats_epsilons[i,j] = get_qtilde(n,alpha,gammastar,epsilon,mstar)
                #print(f"n:{n}, epsilon:{epsilon}, mstar:{mstar}, gammastar:{gammastar}, qhat:{qhats_ns[i,j]:.3f}")
            except:
                qhats_epsilons[i,j] = None
    return qhats_epsilons

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    random.seed(seed)

if __name__ == "__main__":
    sns.set(palette='pastel')
    sns.set_theme(style='white')
    fix_randomness(seed=0)
    # Experimental parameters
    ns = np.logspace(2,4.5,50).astype(int)
    ns_small = [100,1000,10000]
    alpha = 0.1
    epsilons = np.logspace(-2,2,50)
    epsilons_small = [0.1,1,10]
    fname_ns = '.cache/qhats_ns.npy'
    fname_epsilons = '.cache/qhats_epsilons.npy'
    vanilla_conformal = np.array([( (n+1) * (1-alpha) ) / n for n in ns])
    try:
        # load the curves
        qhats_ns = np.load(fname_ns)
        qhats_epsilons = np.load(fname_epsilons)
    except:
        # compute the curves
        qhats_ns = get_qhats_ns(ns, alpha, epsilons_small)
        qhats_epsilons = get_qhats_epsilons(epsilons, alpha, ns_small)
        np.save(fname_ns, qhats_ns)
        np.save(fname_epsilons, qhats_epsilons)
    # plot
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))
    for i in tqdm(range(qhats_ns.shape[0])):
        axs[0].plot(ns,qhats_ns[i,:],label=r"$\epsilon$" + f"={epsilons_small[i]}", linewidth=3, alpha=0.7)
    axs[0].plot(ns, vanilla_conformal, label="nonprivate", c='#ffb347', linestyle='--', linewidth=3, alpha=0.7)
    for i in tqdm(range(qhats_epsilons.shape[0])):
        axs[1].plot(epsilons,qhats_epsilons[i,:],label=r"$n$" + f"={ns_small[i]}", linewidth=3, alpha=0.7)
    sns.despine(top=True, right=True, ax=axs[0])
    sns.despine(top=True, right=True, ax=axs[1])
    axs[0].set_ylim(0.88,1)
    axs[0].axhline(y=1-alpha, c='#999999', linestyle='--', alpha=0.7, label=r'$1-\alpha$')
    axs[0].legend()
    axs[0].set_xlabel('n')
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$\tilde{q}$')
    axs[1].set_ylim(0.88,1)
    axs[1].axhline(y=1-alpha, c='#999999', linestyle='--', alpha=0.7, label=r'$1-\alpha$', linewidth=3)
    axs[1].set_xscale('log')
    axs[1].legend()
    axs[1].set_xlabel(r'$\epsilon$')
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    plt.tight_layout()
    plt.savefig('./outputs/experiment5.pdf')
