import os
import numpy as np
import torch
from torchvision import datasets, transforms as trn
from torch.utils.data import DataLoader

from saige_dataset import *  # Import Severstal, SDI, HBT, DAGM

datasets = {
    "cifar10": "",
    "cifar100": "",
    "svhn": "",
    "tinyimagenet": "",
    "severstal": "",
    "dagm": "",
    "hbt": "",
    "sdi": "",
    # Add new dataset here.
}


def getDataSet(dataset, data_root, split_root, split, batch_size, transform, target_transform=None, num_workers=4, pin_memory=True):
    if split == 'train':
        train = True
    else:
        train = False
    if dataset == 'severstal':
        loader = DataLoader(Severstal(data_root=data_root,
                                      split_root=split_root,
                                      split=split,
                                      transform=transfrom,
                                      target_transform=target_transform),
                            batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Dataset Severstal ready.')
    elif dataset == 'dagm':
        loader = DataLoader(DAGM(data_root=data_root,
                                 split_root=split_root,
                                 split=split,
                                 transform=transfrom,
                                 target_transform=target_transform),
                            batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Dataset DAGM ready.')
    elif dataset == 'hbt':
        loader = DataLoader(HBT(data_root=data_root,
                                split_root=split_root,
                                split=split,
                                transform=transfrom,
                                target_transform=target_transform),
                            batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Dataset HBT ready.')
    elif dataset == 'sdi':
        loader = DataLoader(SDI(data_root=data_root,
                                split_root=split_root,
                                split=split,
                                transform=transfrom,
                                target_transform=target_transform),
                            batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Dataset SDI ready.')
    elif dataset == 'cifar10':
        loader = DataLoader(datasets.CIFAR10(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Dataset CIFAR10 ready.')
    elif dataset == 'cifar100':
        loader = DataLoader(datasets.CIFAR100(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Dataset CIFAR100 ready.')
    elif dataset == 'svhn': # FIX
        loader = DataLoader(datasets.SVHN(root=data_root,
                                            train=train,
                                            download=True,
                                            transform=transform),
                           batch_size=batch_size, shuffle=train,
                            num_workers=num_workers, pin_memory=pin_memory)
        print('Dataset SVHN ready.')
    elif dataset == 'tinyimagenet':
        loader = torch.utils.data.DataLoader(TinyImages(root=data_root,
                                                             transform=transform),
                                                  batch_size=batch_size, 
                                                  shuffle=train, 
                                                  num_workers=num_workers, 
                                                  pin_memory=pin_memory)
    else:
        print("Given dataset {} doesn't exist in implemented dataset.".format(dataset))
        
    return loader