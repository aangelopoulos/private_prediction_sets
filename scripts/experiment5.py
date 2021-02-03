import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.private_conformal_utils import *
import pdb

def get_qhat_n(n, alpha, epsilon, num_replicates):
    gamma = 1/((n * epsilon)**(2/3))
    m = int(get_mstar(n, alpha, epsilon, gamma, num_replicates))
    gamma, _ = get_optimal_gamma(n, alpha, m, epsilon, num_replicates)
    bins = np.linspace(0,1,m)
    qhat = get_qhat(n,alpha,epsilon,gamma,bins,num_replicates)
    return qhat

def get_qhats_ns(ns, alpha, epsilons_small, num_replicates):
    qhats_ns = np.zeros((len(epsilons_small),ns.shape[0]))
    for i in range(len(epsilons_small)):
        for j in range(ns.shape[0]):
            qhats_ns[i,j] = get_qhat_n(ns[j],alpha,epsilons_small[i],num_replicates)
    return qhats_ns

def get_qhats_epsilons(epsilons, alpha, ns_small, num_replicates):
    qhats_ns = np.zeros((len(ns_small),epsilons.shape[0]))
    for i in range(len(ns_small)):
        for j in range(epsilons.shape[0]):
            qhats_ns[i,j] = get_qhat_n(ns_small[i],alpha,epsilons[j],num_replicates)
    return qhats_ns

if __name__ == "__main__":
    sns.set(palette='pastel')
    sns.set_theme(style='white')
    # Experimental parameters
    ns = np.logspace(2,4.5,50).astype(int)
    ns_small = [1000,5000,10000]
    alpha = 0.1
    epsilons = np.logspace(-1,2,50)
    epsilons_small = [1,5,10]
    num_replicates = 100000
    fname_ns = '.cache/qhats_ns.npy'
    fname_epsilons = '.cache/qhats_epsilons.npy'
    vanilla_conformal = np.array([np.ceil( (n+1) * (1-alpha) ) / n for n in ns])
    try:
        # load the curves
        qhats_ns = np.load(fname_ns)
        qhats_epsilons = np.load(fname_epsilons)
    except:
        # compute the curves
        qhats_ns = get_qhats_ns(ns, alpha, epsilons_small, num_replicates)
        qhats_epsilons = get_qhats_epsilons(epsilons, alpha, ns_small, num_replicates)
        np.save(fname_ns, qhats_ns)
        np.save(fname_epsilons, qhats_epsilons)
    # plot
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))
    for i in range(qhats_ns.shape[0]):
        axs[0].plot(ns,qhats_ns[i,:],label=r"$\epsilon$" + f"={epsilons_small[i]}")
    axs[0].plot(ns, vanilla_conformal, label="nonprivate", c='#ffb347', linestyle='--', alpha=0.7)
    for i in range(qhats_epsilons.shape[0]):
        axs[1].plot(epsilons[:-5],qhats_epsilons[i,:-5],label=r"$n$" + f"={ns_small[i]}")
    sns.despine(top=True, right=True, ax=axs[0])
    sns.despine(top=True, right=True, ax=axs[1])
    axs[0].set_ylim(0.88,1)
    axs[0].axhline(y=1-alpha, c='#999999', linestyle='--', alpha=0.7, label=r'$1-\alpha$')
    axs[0].legend()
    axs[0].set_xlabel('n')
    axs[0].set_ylabel(r'$\hat{q}$')
    axs[1].set_ylim(0.88,1)
    axs[1].axhline(y=1-alpha, c='#999999', linestyle='--', alpha=0.7, label=r'$1-\alpha$')
    axs[1].legend()
    axs[1].set_xlabel(r'$\epsilon$')
    plt.tight_layout()
    plt.savefig('./outputs/experiment5.pdf')
