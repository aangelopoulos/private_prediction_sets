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

def experiment(logits_dataset_dict, image_dataset_dict, alpha, epsilon, opt_gamma, num_calib, M, unit, num_replicates_process, batch_size, privatemodel, privateconformal):
    score_bins = np.linspace(0,1,M)
    dataset_precomputed = logits_dataset_dict['val']
    test_targets = [x[1] for x in image_dataset_dict['val']]

    classes_array = ['bacterial pneumonia', 'normal', 'viral pneumonia'] 
    class_idx = np.where(np.array(classes_array) == 'viral pneumonia')[0][0]
    T = platt_logits(dataset_precomputed)
    logits, labels = dataset_precomputed.tensors
    scores = (logits/T.cpu()).softmax(dim=1)

    with torch.no_grad():
        model = get_model(privatemodel, True)

        # do the conformal calibration
        conformal_scores = get_conformal_scores(scores, labels)
        calib_conformal_scores, val_conformal_scores = (1-conformal_scores[0:num_calib], 1-conformal_scores[num_calib:])
        calib_raw_scores, val_raw_scores = (1-scores[0:num_calib], 1-scores[num_calib:])
        
        shat, gamma = get_shat_from_scores_private_opt(calib_conformal_scores, alpha, epsilon, opt_gamma, score_bins, num_replicates_process)

        # give only the dogs where the sets cover the correct answer
        class_selector = np.array(test_targets) == class_idx
        class_selector[0:num_calib] = False # adjust for the calib set.
        class_selector[1-conformal_scores > shat] = False
        selected_images = [image_dataset_dict['val'][i][0] for i in range(len(test_targets)) if class_selector[i]]
        scores = scores[class_selector,:]
        sets = (1-scores < shat)
        sizes = sets.sum(dim=1)

        # find an easy, medium, and hard dog, where the sets are correct
        easyimg_idx = int(np.argmax(np.logical_and(sizes==1, scores.argmax(dim=1)==class_idx)))
        mediumimg_idx = int(np.argmax(sizes==2))
        hardimg_idx = int(np.argmax( sizes == 3 ))

        easyimg = selected_images[easyimg_idx].permute(1,2,0)
        mediumimg = selected_images[mediumimg_idx].permute(1,2,0)
        hardimg = selected_images[hardimg_idx].permute(1,2,0)

        easyperm = np.argsort(scores[easyimg_idx]).flip(dims=(0,))
        mediumperm = np.argsort(scores[mediumimg_idx]).flip(dims=(0,))
        hardperm = np.argsort(scores[hardimg_idx]).flip(dims=(0,))

        easyclasses = [classes_array[easyperm[i]] for i in range(int(sizes[easyimg_idx]))] 
        mediumclasses = [classes_array[mediumperm[i]] for i in range(int(sizes[mediumimg_idx]))] 
        hardclasses = [classes_array[hardperm[i]] for i in range(int(sizes[hardimg_idx]))] 

        plt.imshow(easyimg)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./outputs/three_covid/easy_covid.png', bbox_inches='tight')
        print(f"easy classes: {easyclasses}")
        plt.imshow(mediumimg)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./outputs/three_covid/medium_covid.png', bbox_inches='tight')
        print(f"medium classes: {mediumclasses}")
        plt.imshow(hardimg)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./outputs/three_covid/hard_covid.png', bbox_inches='tight')
        print(f"hard classes: {hardclasses}")

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

    alpha = 0.1
    epsilon = 8 # epsilon of the trained model 
    opt_gamma = np.logspace(-4,-0.5,50)
    num_calib = 1000 
    num_val = 500
    num_trials = 100 
    num_replicates_process =100000
    privatemodel = False
    privateconformal = True
    
    unit = int(np.floor(np.sqrt(num_calib)))
    M = get_mstar(num_calib, alpha, epsilon, 0.05, num_replicates_process) 
    class_to_find = 'viral'
    print(M)

    # Top level data directory. Here we assume the format of the directory conforms 
    #   to the ImageFolder structure
    data_dir = "./data/imagefolder"
    datasetname = 'xray'
    logits_dataset_dict, image_dataset_dict = get_logits_dataset(privatemodel, datasetname, data_dir, num_calib, num_val, seed=0, cache='./.cache/')
    experiment(logits_dataset_dict, image_dataset_dict, alpha, epsilon, opt_gamma, num_calib, M, unit, num_replicates_process, batch_size=128, privatemodel=privatemodel, privateconformal=privateconformal)
