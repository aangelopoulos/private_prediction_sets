import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import torch
import torchvision as tv
import argparse
import time
import numpy as np
from scipy.stats import binom
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from utils import *
import seaborn as sns
from core.concentration import *
from core.private_conformal_utils import *
import pdb

def get_conformal_scores(scores, labels):
    conformal_scores = torch.tensor([scores[i,labels[i]] for i in range(scores.shape[0])]) 
    return conformal_scores 

def get_shat_from_scores(scores, alpha):
    return np.quantile(scores,1-alpha)

def get_shat_from_scores_private_opt(scores, alpha, epsilon, opt_gamma, score_bins, num_replicates_process):
    best_gamma = opt_gamma[0] # dummy initialization
    best_shat = scores.max()
    for i in range(opt_gamma.shape[0]):
        gamma = opt_gamma[i]
        shat = get_private_quantile(scores, alpha, epsilon, gamma, score_bins, num_replicates_process)
        if shat <= best_shat:
            best_shat = shat
            best_gamma = gamma 
    return best_shat, best_gamma

def trial_precomputed(conformal_scores, raw_scores, alpha, epsilon, opt_gamma, score_bins, num_replicates_process, num_calib, batch_size, privateconformal):
    total=conformal_scores.shape[0]
    perm = torch.randperm(conformal_scores.shape[0])
    conformal_scores = conformal_scores[perm]
    raw_scores = raw_scores[perm]
    calib_conformal_scores, val_conformal_scores = (1-conformal_scores[0:num_calib], 1-conformal_scores[num_calib:])
    calib_raw_scores, val_raw_scores = (1-raw_scores[0:num_calib], 1-raw_scores[num_calib:])
    
    if privateconformal:
        shat, gamma = get_shat_from_scores_private_opt(calib_conformal_scores, alpha, epsilon, opt_gamma, score_bins, num_replicates_process)
    else:
        gamma = 0
        shat = get_shat_from_scores(calib_conformal_scores, alpha)

    corrects = (val_conformal_scores) < shat 
    sizes = ((val_raw_scores) < shat).sum(dim=1)

    return corrects.float().mean().item(), torch.tensor(sizes), shat, gamma

def plot_histograms(df_list,alpha,M,unit,num_calib,privatemodel,privateconformal,num_trials):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    mincvg = min([df['coverage'].min() for df in df_list])
    maxcvg = max([df['coverage'].max() for df in df_list])

    cvg_bins = np.arange(1-alpha-0.02,1.01,0.005)#None #np.arange(mincvg, maxcvg, 0.001) 
    
    if privateconformal:
        for i in range(len(df_list)):
            df = df_list[i]
            print(f"alpha:{alpha}, epsilon:{epsilon}, coverage:{np.median(df.coverage)}")
            # Use the same binning for everybody 
            weights = np.ones((len(df),))/len(df)
            axs[0].hist(np.array(df['coverage'].tolist()), cvg_bins, alpha=0.7, density=False, weights=weights)

            # Sizes will be 10 times as big as risk, since we pool it over runs.
            sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
            d = np.diff(np.unique(sizes)).min()
            lofb = sizes.min() - float(d)/2
            rolb = sizes.max() + float(d)/2
            weights = np.ones_like(sizes)/sizes.shape[0]
            #axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label=f"M={M/unit:.2f}" + r"$\times \sqrt{n}$", alpha=0.7, density=True)
            axs[1].hist(sizes, label=f"M={M/unit:.2f}" + r"$\times \sqrt{n}$", alpha=0.7, density=False, weights=weights)
    else:
        df = df_list[0]
        axs[0].hist(np.array(df['coverage'].tolist()), cvg_bins, alpha=0.7, density=False)
        weights = np.ones((len(df),))/len(df)
        # Sizes will be 10 times as big as risk, since we pool it over runs.
        sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
        d = np.diff(np.unique(sizes)).min()
        lofb = sizes.min() - float(d)/2
        rolb = sizes.max() + float(d)/2
        weights = np.ones_like(sizes)/sizes.shape[0]
        #axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label=f"M={M/unit:.2f}" + r"$\times \sqrt{n}$", alpha=0.7, density=True)
        axs[1].hist(sizes, alpha=0.7, density=False)

    axs[0].set_xlabel('coverage')
    #axs[0].locator_params(axis='x', nbins=5)
    axs[0].set_xlim([1-alpha-0.02, 1.01])
    #axs[0].set_ylim([0,num_trials])
    #axs[0].set_yscale('log')
    axs[0].set_ylabel('probability')
    #axs[0].set_yticks([0,100])
    axs[0].axvline(x=1-alpha,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    axs[1].set_xlim([0.5,10.5])
    #axs[1].set_yscale('log')
    #axs[1].set_xscale('log')

    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    plt.tight_layout()
    privatemodel_str = 'privatemodel' if privatemodel else 'nonprivatemodel'
    privateconformal_str = 'privateconformal' if privateconformal else 'nonprivateconformal'
    plt.savefig( f'outputs/histograms/experiment1.pdf')

def experiment(alpha, epsilon, opt_gamma, num_calib, M, unit, num_replicates_process, batch_size, cifar10_root, privatemodel, privateconformal):
    df_list = []
    score_bins = np.linspace(0,1,M)
    fname = f'.cache/opt_{privatemodel}_{privateconformal}_{alpha}_{epsilon}_{opt_gamma[0]}_{opt_gamma[-1]}_{num_calib}_{M}bins_dataframe.pkl'

    df = pd.DataFrame(columns = ["$\\hat{s}$","coverage","sizes","$\\alpha$","$\\epsilon$", "$\\gamma$"])
    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError:
        dataset_precomputed = get_logits_dataset(privatemodel, 'CIFAR10', cifar10_root)
        print('Dataset loaded')
        
        classes_array = get_cifar10_classes()
        T = platt_logits(dataset_precomputed)
        logits, labels = dataset_precomputed.tensors
        scores = (logits/T.cpu()).softmax(dim=1)

        with torch.no_grad():
            conformal_scores = get_conformal_scores(scores, labels)
            local_df_list = []
            for i in tqdm(range(num_trials)):
                cvg, szs, shat, gamma = trial_precomputed(conformal_scores, scores, alpha, epsilon, opt_gamma, score_bins, num_replicates_process, num_calib, batch_size, privateconformal)
                dict_local = {"$\\hat{s}$": shat,
                                "coverage": cvg,
                                "sizes": [szs],
                                "$\\alpha$": alpha,
                                "$\\epsilon$": epsilon,
                                "$\\gamma$": gamma
                             }
                df_local = pd.DataFrame(dict_local)
                local_df_list = local_df_list + [df_local]
            df = pd.concat(local_df_list, axis=0, ignore_index=True)
            df.to_pickle(fname)

    df_list = df_list + [df]

    plot_histograms(df_list,alpha,M,unit,num_calib,privatemodel,privateconformal,num_trials)

def platt_logits(calib_dataset, max_iters=10, lr=0.01, epsilon=0.01):
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=1024, shuffle=False, pin_memory=True) 
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    fix_randomness(seed=0)

    cifar10_root = './data/cifar10'
    privateconformal = True 
    privatemodel = True

    alpha = 0.1
    epsilon = 3.29 # epsilon of the trained model 
    opt_gamma = np.logspace(-4,-0.5,50)
    num_calib = 5000 
    num_trials = 100 
    num_replicates_process =100000
    
    unit = int(np.floor(np.sqrt(num_calib)))
    M = get_mstar(num_calib, alpha, epsilon, 0.05, num_replicates_process) #np.floor(5*unit).astype(int)
    print(M)

    experiment(alpha, epsilon, opt_gamma, num_calib, M, unit, num_replicates_process, batch_size=128, cifar10_root=cifar10_root, privatemodel=privatemodel, privateconformal=privateconformal)
