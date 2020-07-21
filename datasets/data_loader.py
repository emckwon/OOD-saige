import os
import numpy as np
import torch
from torchvision import datasets, transforms as trn
from torch.utils.data import DataLoader

from saige_dataset import SaigeDataset


#def getDataset(dataset, data_root, split_root, split, batch_size, transform, targets, num_workers=4, pin_memory=True):
    
def getDataLoader(ds_cfg, dl_cfg, split)

    if split == 'train':
        train = True
        transform = ds_cfg['train_transform']
    else:
        train = False
        transform = ds_cfg['valid_transform']
        
    if dataset in ['Severstal', 'DAGM', 'HBT/LAMI', 'HBT/NUDE', 'SDI/34Ah',
            'SDI/37Ah', 'SDI/60Ah']:
        loader = DataLoader(SaigeDataset(data_root=ds_cfg['data_root'],
                                         split_root=ds_ccfg['split_root'],
                                         dataset=ds_cfg['dataset'],
                                         split=split,
                                         transform=transform,
                                         targets=ds_cfg['targets']),
                            batch_size=dl_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
    elif dataset == 'cifar10':
        loader = DataLoader(datasets.CIFAR10(root=ds_cfg['data_root'],
                                             train=train,
                                             download=True,
                                             transform=transform),
                            batch_size=dl_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset CIFAR10 ready.')
    elif dataset == 'cifar100':
        loader = DataLoader(datasets.CIFAR100(root=ds_cfg['data_root'],
                                              train=train,
                                              download=True,
                                              transform=transform),
                            batch_size=dl_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset CIFAR100 ready.')
    elif dataset == 'svhn': # FIX
        loader = DataLoader(datasets.SVHN(root=ds_cfg['data_root'],
                                          train=train,
                                          download=True,
                                          transform=transform),
                            batch_size=dl_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset SVHN ready.')
    elif dataset == 'tinyimagenet':
        loader = torch.utils.data.DataLoader(TinyImages(root=data_root,
                                                        transform=transform),
                                             batch_size=dl_cfg['batch_size'],
                                             shuffle=train,
                                             num_workers=dl_cfg['num_workers'],
                                             pin_memory=dl_cfg['pin_memory'])
    else:
        print("Given dataset {} doesn't exist in implemented dataset.".format(dataset))
        
    return loader