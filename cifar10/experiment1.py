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
from core.private_conformal_utils import *

def get_conformal_scores(scores, labels):
    conformal_scores = torch.tensor([scores[i,labels[i]] for i in range(scores.shape[0])])
    return conformal_scores

def get_shat_from_scores(scores, alpha):
    return np.quantile(scores,1-alpha)

def get_shat_from_scores_private(scores, alpha, epsilon, gamma, score_bins):
    shat = get_private_quantile(scores, alpha, epsilon, gamma, score_bins)
    return shat

def trial_precomputed(conformal_scores, raw_scores, alpha, epsilon, gamma, score_bins, num_calib, privateconformal):
    total=conformal_scores.shape[0]
    perm = torch.randperm(conformal_scores.shape[0])
    conformal_scores = conformal_scores[perm]
    raw_scores = raw_scores[perm]
    calib_conformal_scores, val_conformal_scores = (1-conformal_scores[0:num_calib], 1-conformal_scores[num_calib:])
    calib_raw_scores, val_raw_scores = (1-raw_scores[0:num_calib], 1-raw_scores[num_calib:])

    if privateconformal:
        shat = get_shat_from_scores_private(calib_conformal_scores, alpha, epsilon, gamma, score_bins)
    else:
        gamma = 0
        shat = get_shat_from_scores(calib_conformal_scores, alpha)

    corrects = (val_conformal_scores) < shat
    sizes = ((val_raw_scores) < shat).sum(dim=1)

    return corrects.float().mean().item(), torch.tensor(sizes), shat

def plot_histograms(df_list,alpha):
    fig_cvg, axs_cvg = plt.subplots(nrows=2,ncols=2,figsize=(6,6))
    fig_sz, axs_sz = plt.subplots(nrows=2,ncols=2,figsize=(6,6))

    mincvg = min([df['coverage'].min() for df in df_list])
    maxcvg = max([df['coverage'].max() for df in df_list])

    cvg_bins = np.arange(1-alpha-0.02,1.01,0.005)

    for i in range(len(df_list)):
        df = df_list[i]
        print(f"alpha:{alpha}, epsilon:{epsilon}, coverage:{np.median(df.coverage)}")
        # Use the same binning for everybody
        weights = np.ones((len(df),))/len(df)
        axs_cvg[i % 2, i // 2].hist(np.array(df['coverage'].tolist()), cvg_bins, alpha=0.7, density=False, weights=weights)
        axs_cvg[i % 2, i // 2].set_xlim([1-alpha-0.02, 1.01])
        axs_cvg[i % 2, i // 2].axvline(x=1-alpha,c='#999999',linestyle='--',alpha=0.7)

        # Sizes will be 10 times as big as risk, since we pool it over runs.
        sizes = torch.cat(df['sizes'].tolist(),dim=0).numpy()
        d = np.diff(np.unique(sizes)).min()
        lofb = sizes.min() - float(d)/2
        rolb = 11.5#sizes.max() + float(d)/2
        size_bins = np.arange(lofb,rolb,d)

        weights = np.ones_like(sizes)/sizes.shape[0]
        axs_sz[i % 2, i // 2].hist(sizes, size_bins, alpha=0.7, density=False, weights=weights)
        axs_sz[i % 2, i // 2].set_xlim([0.5,10.5])

    for i in range(2):
        for j in range(2):
            sns.despine(ax=axs_cvg[i,j],top=True,right=True)
            sns.despine(ax=axs_sz[i,j],top=True,right=True)
            axs_cvg[i,j].locator_params(axis='y',nbins=4)
            axs_sz[i,j].locator_params(axis='y',nbins=4)

            if i == 0 and j == 0:
                axs_cvg[i,j].set_title('no')
                axs_cvg[i,j].set_ylabel('no')
                axs_sz[i,j].set_title('no')
                axs_sz[i,j].set_ylabel('no')

            if j == 0 and i == 1:
                axs_cvg[i,j].set_ylabel('yes')
                axs_sz[i,j].set_ylabel('yes')

            if j == 1 and i == 0:
                axs_cvg[i,j].set_title('yes')
                axs_sz[i,j].set_title('yes')

    plt.figure(fig_cvg.number)
    major_fontsize = 14
    plt.tight_layout(rect=[0.05,0.05,0.95,0.95])
    plt.text(1.05,1.2,'private conformal', horizontalalignment='center', verticalalignment='top', transform=axs_cvg[0,0].transAxes, fontsize=major_fontsize)
    plt.text(-0.4,-0.2,'private model', horizontalalignment='center', verticalalignment='top', transform=axs_cvg[0,0].transAxes, rotation=90, rotation_mode='anchor', fontsize=major_fontsize)
    plt.text(1.1,-0.2,'coverage', horizontalalignment='center', verticalalignment='top', transform=axs_cvg[1,0].transAxes, fontsize=major_fontsize)
    plt.savefig('outputs/histograms/experiment1_coverage.pdf')
    plt.figure(fig_sz.number)
    plt.tight_layout(rect=[0.05,0.05,0.95,0.95])
    plt.text(1.1,1.2,'private conformal', horizontalalignment='center', verticalalignment='top', transform=axs_sz[0,0].transAxes, fontsize=major_fontsize)
    plt.text(-0.5,-0.2,'private model', horizontalalignment='center', verticalalignment='top', transform=axs_sz[0,0].transAxes, rotation=90, rotation_mode='anchor', fontsize=major_fontsize)
    plt.text(1.1,-0.2,'size', horizontalalignment='center', verticalalignment='top', transform=axs_sz[1,0].transAxes, fontsize=major_fontsize)
    plt.savefig('outputs/histograms/experiment1_size.pdf')

def experiment(alpha, epsilon, gamma, num_calib, m, cifar10_root, privatemodel, privateconformal):
    df_list = []
    score_bins = np.linspace(0,1,m)
    fname = f'.cache/opt_{privatemodel}_{privateconformal}_{alpha}_{epsilon}_{num_calib}_{m}bins_dataframe.pkl'

    df = pd.DataFrame(columns = ["$\\hat{s}$","coverage","sizes","$\\alpha$","$\\epsilon$"])
    try:
        df = pd.read_pickle(fname)
    except FileNotFoundError:
        dataset_precomputed = get_logits_dataset(privatemodel, 'CIFAR10', cifar10_root)
        print('Dataset loaded')

        classes_array = get_cifar10_classes()
        T = platt_logits(dataset_precomputed)
        logits, labels = dataset_precomputed.tensors
        scores = (logits/T.cpu()).softmax(dim=1)

        accuracy = (scores.argmax(dim=1) == labels).float().mean()
        print(f"Private model: {privatemodel}, Accuracy: {accuracy}")

        with torch.no_grad():
            conformal_scores = get_conformal_scores(scores, labels)
            local_df_list = []
            for i in tqdm(range(num_trials)):
                cvg, szs, shat = trial_precomputed(conformal_scores, scores, alpha, epsilon, gamma, score_bins, num_calib, privateconformal)
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
    return df

def platt_logits(calib_dataset, max_iters=10, lr=0.01, epsilon=0.01):
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=1024, shuffle=False, pin_memory=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nll_criterion = nn.CrossEntropyLoss().to(device)

    T = nn.Parameter(torch.Tensor([1.3]).to(device))

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.to(device)
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().to(device))
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
    privateconformals = [False, True]
    privatemodels = [False, True]

    alpha = 0.1
    epsilon = 8 # epsilon of the trained model
    num_calib = 5000
    num_trials = 100
    mstar, gammastar = get_optimal_gamma_m(num_calib, alpha, epsilon)

    df_list = []
    for privateconformal in privateconformals:
        for privatemodel in privatemodels:
            df_list = df_list + [experiment(alpha, epsilon, gammastar, num_calib, mstar, cifar10_root=cifar10_root, privatemodel=privatemodel, privateconformal=privateconformal)]
    plot_histograms(df_list,alpha)
