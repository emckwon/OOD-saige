import torch
from torchvision import datasets, transforms as trn
from torch.utils.data import DataLoader

import os
import numpy as np
import sys


# in_dist_dataset = {
#     "cifar10": "",
#     "cifar100": "",
#     "svhn": "",
    
#     # Add new in-distribution dataset here.
    
# }
# out_dist_dataset = {
#     "cifar10": "",
#     "cifar100": "",
#     "svhn": "",
#     "tinyimagenet": "",
    
#     # Add new out-distribution dataset here
# }


def getInDistDataSet(dataset, data_root, train, batch_size, image_size, transform=None, num_workers=4, pin_memory=True):
    if dataset == 'cifar10':
        loader = DataLoader(datasets.CIFAR10(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('In-distribution dataset CIFAR10 ready.')
    elif dataset == 'cifar100':
        loader = DataLoader(datasets.CIFAR100(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('In-distribution dataset CIFAR100 ready.')
    elif dataset == 'svhn': # FIX
        loader = DataLoader(datasets.SVHN(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('In-distribution dataset SVHN ready.')
    else:
        print("Given dataset {} doesn't exist in in_dist_dataset.".format(dataset))
        
    return loader


def getOutDistDataset(dataset, data_root, train, batch_size, image_size, transform=None, num_workers=4, pin_memory=True)
    if dataset == 'cifar10':
        loader = DataLoader(datasets.CIFAR10(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Out-distribution dataset CIFAR10 ready.')
    elif dataset == 'cifar100':
        loader = DataLoader(datasets.CIFAR100(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('In-distribution dataset CIFAR100 ready.')
    elif dataset == 'svhn':
        loader = DataLoader(datasets.SVHN(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('In-distribution dataset SVHN ready.')
    elif dataset == 'tinyimagenet':
        loader = torch.utils.data.DataLoader(TinyImages(root=data_root,
                                                             transform=transform),
                                                  batch_size=batch_size, 
                                                  shuffle=train, 
                                                  num_workers=num_workers, 
                                                  pin_memory=pin_memory)
    else:
        print("Given dataset {} doesn't exist in out_dist_dataset.".format(dataset))
        
    return loader