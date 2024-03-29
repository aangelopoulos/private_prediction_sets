import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import random
import pandas as pd
import pdb
from torchvision.datasets import CIFAR10
from model_class import convnet
from module_modification import convert_batchnorm_modules

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

def get_model(private=True, cache= dirname + '/.cache/'):
    model = convert_batchnorm_modules(torchvision.models.resnet18(num_classes=10))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if private:
        data = torch.load('./models/privatemodel_best.pth.tar', map_location=device)
    else:
        data = torch.load('./models/nonprivatemodel_best.pth.tar', map_location=device)

    model.load_state_dict(data['state_dict'])
    model.to(device)
    model.eval()

    return model

# Computes logits and targets from a model and loader
def get_logits_targets(model, loader):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logits = torch.zeros((len(loader.dataset), 10)) # 10 classes in CIFAR.
    labels = torch.zeros((len(loader.dataset),))
    model = model.to(device)
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.to(device)).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())
    return dataset_logits

def get_logits_dataset(private, datasetname, datasetpath, cache= dirname + '/.cache/'):
    fname = cache + datasetname + '/' + 'private' + '.pkl'  if private else cache + datasetname + '/nonprivate.pkl'

    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    # Else we will load our model, run it on the dataset, and save/return the output.
    model = get_model(private)

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    test_transform = transforms.Compose(normalize)
    test_dataset = CIFAR10(
        root=datasetpath, train=False, download=True, transform=test_transform
    )

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )

    # Get the logits and targets
    dataset_logits = get_logits_targets(model, loader)

    # Save the dataset
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as handle:
        pickle.dump(dataset_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_logits

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
