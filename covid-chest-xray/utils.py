import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import random
import pandas as pd
import pdb
from torchvision.datasets import CIFAR10

dirname = str(pathlib.Path(__file__).parent.absolute())

def sort_sum(scores):
    I = scores.argsort(axis=1)[:,::-1]
    ordered = np.sort(scores,axis=1)[:,::-1]
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def validate(val_loader, model, losses, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        risks = AverageMeter('empirical losses')
        sizes = AverageMeter('RAPS size')
        sizes_arr = []
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            # measure accuracy and record loss
            risk, size_arr = risk_size(S, target, losses)


            # Update meters
            risks.update(risk, n=x.shape[0])
            sizes.update(size_arr.mean(), n=x.shape[0])
            sizes_arr = sizes_arr + [size_arr]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Risks: {risks.val:.3f} ({risks.avg:.3f}) | Sizes: {sizes.val:.3f} ({sizes.avg:.3f}) ', end='')
    if print_bool:
        print('') #Endline

    return risks.avg, sizes_arr 

def risk_size(S,targets, losses):
    risk = 0
    size_arr = np.zeros((targets.shape[0],))
    for i in range(targets.shape[0]):
        if (targets[i].item() not in S[i]):
            risk += losses[targets[i].item()] 
        size_arr[i] = S[i].shape[0]
    return float(risk)/targets.shape[0], size_arr 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).cuda()
    targets = torch.cat([torch.Tensor([int(x[1])]) for x in data], dim=0).long()
    return imgs, targets

def split2ImageFolder(path, transform, n1, n2):
    dataset = torchvision.datasets.ImageFolder(path, transform)
    data1, data2 = torch.utils.data.random_split(dataset, [n1, len(dataset)-n1])
    data2, _ = torch.utils.data.random_split(data2, [n2, len(dataset)-n1-n2])
    return data1, data2

def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    return data1, data2

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(private=False, feature_extract=True, cache= dirname + '/.cache/'):
    model_ft = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)

    data = torch.load('./.cache/nonprivatemodel_best.pth.tar')

    model_ft.load_state_dict(data)
    model_ft.cuda()
    model_ft.eval()

    return model_ft

# Computes logits and targets from a model and loader
def get_logits_targets(model, loader):
    logits = torch.zeros((len(loader.dataset), 3)) # 3 classes in XRAY.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.cuda()).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits

def get_dataset_shuffle_split(datasetpath, num_calib, num_val, seed):
    # Create training and validation datasets
    input_size = 224
    batch_size = 256

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")
    fix_randomness(seed)
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(datasetpath, x), data_transforms[x]) for x in ['train', 'val']}
    temp = torch.utils.data.ConcatDataset([image_datasets['train'],image_datasets['val']])
    image_datasets['train'], image_datasets['val'] = torch.utils.data.random_split(temp,[len(temp)-num_calib-num_val,num_calib+num_val])
    return image_datasets

def get_logits_dataset(private, datasetname, datasetpath, num_calib, num_val, seed, cache= dirname + '/.cache/'):
    fname = cache + datasetname + '/' + 'private' + '.pkl'  if private else cache + datasetname + '/nonprivate.pkl'
    batch_size = 256

    image_datasets = get_dataset_shuffle_split(datasetpath, num_calib, num_val, seed)
    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle), image_datasets

    # Else we will load our model, run it on the dataset, and save/return the output.
    model = get_model(private, True)

    # get the datasets and loaders
    image_datasets = get_dataset_shuffle_split(datasetpath, num_calib, num_val, seed)

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Get the logits and targets
    dataset_logits_dict = {x: get_logits_targets(model, dataloaders_dict[x]) for x in ['train','val']}

    # Save the dataset 
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as handle:
        pickle.dump(dataset_logits_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_logits_dict, image_datasets

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def get_cifar10_classes():
    return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] 

def get_metrics_precomputed(est_labels,labels,losses,num_classes):
    labels = torch.nn.functional.one_hot(labels,num_classes)
    empirical_losses = (losses.view(1,-1) * (labels * (1-est_labels))).sum(dim=1)
    sizes = est_labels.sum(dim=1)
    return empirical_losses, sizes 
