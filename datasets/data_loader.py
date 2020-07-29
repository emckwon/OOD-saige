import os
import numpy as np
import torch
from torchvision import datasets, transforms as trn
from torch.utils.data import DataLoader

from datasets.tinyimages_80mn_loader import TinyImages
from datasets.saige_dataset import SaigeDataset

    
def getDataLoader(ds_cfg, dl_cfg, split):
    if split == 'train':
        train = True
        transform = ds_cfg['train_transform']
    else:
        train = False
        transform = ds_cfg['valid_transform']
        
    if ds_cfg['split'] == 'train':
        split = 'train'
    elif ds_cfg['split'] == 'valid':
        split = 'valid'
    elif ds_cfg['split'] == 'test':
        split = 'test'
    else: 
        pass
    
    if ds_cfg['dataset'] in ['Severstal', 'DAGM', 'HBT/LAMI', 'HBT/NUDE', 'SDI/34Ah',
            'SDI/37Ah', 'SDI/60Ah']:
        loader = DataLoader(SaigeDataset(data_root=ds_cfg['data_root'],
                                         split_root=ds_cfg['split_root'],
                                         dataset=ds_cfg['dataset'],
                                         split=split,
                                         transform=transform,
                                         targets=ds_cfg['targets']),
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
    elif ds_cfg['dataset'] == 'cifar10':
        loader = DataLoader(datasets.CIFAR10(root=ds_cfg['data_root'],
                                             train=train,
                                             download=True,
                                             transform=transform),
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset CIFAR10 ready.')
    elif ds_cfg['dataset'] == 'cifar100':
        loader = DataLoader(datasets.CIFAR100(root=ds_cfg['data_root'],
                                              train=train,
                                              download=True,
                                              transform=transform),
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset CIFAR100 ready.')
    elif ds_cfg['dataset'] == 'svhn': # FIX
        loader = DataLoader(datasets.SVHN(root=ds_cfg['data_root'],
                                          split="train" if split == "train" else "test",
                                          download=True,
                                          transform=transform),
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset SVHN ready.')
    elif ds_cfg['dataset'] == 'tinyimagenet':
        loader = torch.utils.data.DataLoader(TinyImages(root=ds_cfg['data_root'],
                                                        transform=transform),
                                             batch_size=dl_cfg['batch_size'],
                                             shuffle=train,
                                             num_workers=ds_cfg['num_workers'],
                                             pin_memory=dl_cfg['pin_memory'])
        print('Dataset Tinyimagenet ready.')
    else:
        raise NotImplementedError(
            print("Given dataset {} doesn't exist in implemented dataset.".format(dataset))
        )

        
    return loader