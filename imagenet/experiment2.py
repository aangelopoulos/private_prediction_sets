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

def get_shat_from_scores_private(scores, alpha, epsilon, gamma, score_bins, num_replicates_process):
    shat = get_private_quantile(scores, alpha, epsilon, gamma, score_bins, num_replicates_process)
    return shat 

def trial_precomputed(conformal_scores, raw_scores, alpha, epsilon, gamma, score_bins, num_replicates_process, num_calib, batch_size):
    total=conformal_scores.shape[0]
    perm = torch.randperm(conformal_scores.shape[0])
    conformal_scores = conformal_scores[perm]
    raw_scores = raw_scores[perm]
    calib_conformal_scores, val_conformal_scores = (1-conformal_scores[0:num_calib], 1-conformal_scores[num_calib:])
    calib_raw_scores, val_raw_scores = (1-raw_scores[0:num_calib], 1-raw_scores[num_calib:])

    shat = get_shat_from_scores_private(calib_conformal_scores, alpha, epsilon, gamma, score_bins, num_replicates_process)

    corrects = (val_conformal_scores) < shat 
    sizes = ((val_raw_scores) < shat).sum(dim=1)

    return corrects.float().mean().item(), torch.tensor(sizes), shat

def plot_histograms(df_list,alpha,Ms,unit,num_calib):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))

    mincvg = min([df['coverage'].min() for df in df_list])
    maxcvg = min([df['coverage'].max() for df in df_list])

    cvg_bins = None 
    
    for i in range(len(df_list)):
        df = df_list[i]
        M = Ms[i]
        print(f"alpha:{alpha}, epsilon:{epsilon}, coverage:{np.median(df.coverage)}")
        # Use the same binning for everybody 
        weights = np.ones((len(df),))/len(df)
        axs[0].hist(np.array(df['coverage'].tolist()), cvg_bins, alpha=0.7, weights=weights)

        # Sizes will be 10 times as big as risk, since we pool it over runs.
        sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
        d = np.diff(np.unique(sizes)).min()
        lofb = sizes.min() - float(d)/2
        rolb = sizes.max() + float(d)/2
        weights = np.ones_like(sizes)/sizes.shape[0]
        mstar_str = ''
        if i == len(df_list)-1:
            mstar_str = r'$^*$'
        axs[1].hist(sizes, np.arange(lofb,rolb+d, d), label=f"m{mstar_str}={np.round(M/unit,2):.2f}" + r"$\times n\epsilon$", alpha=0.7, weights=weights)
    
    axs[0].set_xlabel('coverage')
    axs[0].set_ylabel('density')
    axs[0].set_xlim([1-alpha-0.002,None])
    axs[0].axvline(x=1-alpha,c='#999999',linestyle='--',alpha=0.7)
    axs[1].set_xlabel('size')
    axs[1].legend()
    axs[1].set_xlim([0.5,None])
    sns.despine(ax=axs[0],top=True,right=True)
    sns.despine(ax=axs[1],top=True,right=True)
    plt.tight_layout()
    plt.savefig( 'outputs/histograms/experiment2.pdf')

def experiment(alpha, epsilon, num_calib, Ms, unit, num_replicates_process, batch_size, imagenet_val_dir):
    df_list = []
    for M in Ms:
        gamma, _ = get_optimal_gamma(num_calib,alpha,int(M),epsilon,num_replicates_process)
        score_bins = np.linspace(0,1,M)
        fname = f'.cache/opt_{alpha}_{epsilon}_{num_calib}_{M}bins_dataframe.pkl'

        df = pd.DataFrame(columns = ["$\\hat{s}$","coverage","sizes","$\\alpha$","$\\epsilon$" ])
        try:
            df = pd.read_pickle(fname)
        except FileNotFoundError:
            dataset_precomputed = get_logits_dataset('ResNet152', 'Imagenet', imagenet_val_dir)
            print('Dataset loaded')
            
            classes_array = get_imagenet_classes()
            T = platt_logits(dataset_precomputed)
            
            logits, labels = dataset_precomputed.tensors
            scores = (logits/T.cpu()).softmax(dim=1)

            with torch.no_grad():
                conformal_scores = get_conformal_scores(scores, labels)
                local_df_list = []
                for i in tqdm(range(num_trials)):
                    cvg, szs, shat = trial_precomputed(conformal_scores, scores, alpha, epsilon, gamma, score_bins, num_replicates_process, num_calib, batch_size)
                    dict_local = {"$\\hat{s}$": shat,
                                    "coverage": cvg,
                                    "sizes": [szs],
                                    "$\\alpha$": alpha,
                                    "$\\epsilon$": epsilon
                                 }
                    df_local = pd.DataFrame(dict_local)
                    local_df_list = local_df_list + [df_local]
                df = pd.concat(local_df_list, axis=0, ignore_index=True)
                df.to_pickle(fname)

        df_list = df_list + [df]

    plot_histograms(df_list,alpha,Ms,unit,num_calib)

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

    imagenet_val_dir = '/scratch/group/ilsvrc/val' # TODO: Put your imagenet directory here

    alpha = 0.1
    epsilon = 1
    num_calib = 30000 
    num_trials = 100 
    num_replicates_process =100000
    
    unit = (num_calib * epsilon)

    Mstar = get_mstar(num_calib, alpha, epsilon, 0.04, num_replicates_process) # max number of bins
    Ms = np.floor(np.array([0.05*unit,0.1*unit,1*unit,10*unit,Mstar])).astype(int)

    experiment(alpha, epsilon, num_calib, Ms, unit, num_replicates_process, batch_size=128, imagenet_val_dir=imagenet_val_dir)
