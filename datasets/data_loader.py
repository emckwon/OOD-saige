import os
import numpy as np
import torch
from torchvision import datasets, transforms as trn
from torch.utils.data import DataLoader

from datasets.tinyimages_80mn_loader import TinyImages
from datasets.saige_dataset import SaigeDataset, SaigeDataset2
from datasets.PerturbDataset import PerturbDataset

    
def getDataLoader(ds_cfg, dl_cfg, split):
    if split == 'train':
        train = True
        transform = ds_cfg['train_transform']
    else:
        train = False
        transform = ds_cfg['valid_transform']
        
    if 'split' in ds_cfg.keys() and ds_cfg['split'] == 'train':
        split = 'train'
    elif 'split' in ds_cfg.keys() and ds_cfg['split'] == 'valid':
        split = 'valid'
    elif 'split' in ds_cfg.keys() and ds_cfg['split'] == 'test':
        split = 'test'
    else: 
        pass
    
    if ds_cfg['dataset'] in ['Severstal', 'DAGM', 'HBT/LAMI', 'HBT/NUDE', 'SDI/34Ah',
            'SDI/37Ah', 'SDI/60Ah']:
        dataset = SaigeDataset(data_root=ds_cfg['data_root'],
                                         split_root=ds_cfg['split_root'],
                                         dataset=ds_cfg['dataset'],
                                         split=split,
                                         transform=transform,
                                         targets=ds_cfg['targets'])
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)
                               
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
    elif ds_cfg['dataset'] == 'Daeduck':
        dataset = SaigeDataset2(data_root=ds_cfg['data_root'],
                                         split_root=ds_cfg['split_root'],
                                         dataset=ds_cfg['dataset'],
                                         split=split,
                                         transform=transform,
                                         targets=ds_cfg['targets'])
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)
                               
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset {} ready.'.format(ds_cfg['dataset']))
        
    elif ds_cfg['dataset'] == 'cifar10':
        dataset = datasets.CIFAR10(root=ds_cfg['data_root'],
                                             train=train,
                                             download=True,
                                             transform=transform)
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)
            
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset CIFAR10 ready.')
    elif ds_cfg['dataset'] == 'cifar100':
        dataset = datasets.CIFAR100(root=ds_cfg['data_root'],
                                              train=train,
                                              download=True,
                                              transform=transform)
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)
            
        loader = DataLoader(dataset,
                            batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset CIFAR100 ready.')
    elif ds_cfg['dataset'] == 'svhn': # FIX
        dataset = datasets.SVHN(root=ds_cfg['data_root'],
                                          split="train" if split == "train" else "test",
                                          download=True,
                                          transform=transform)
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)        
        
        loader = DataLoader(dataset,batch_size=ds_cfg['batch_size'], shuffle=train,
                            num_workers=dl_cfg['num_workers'], pin_memory=dl_cfg['pin_memory'])
        print('Dataset SVHN ready.')
    elif ds_cfg['dataset'] == 'tinyimagenet':
        dataset = TinyImages(root=ds_cfg['data_root'], transform=transform)
        
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)     
            
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=ds_cfg['batch_size'],
                                             shuffle=train,
                                             num_workers=dl_cfg['num_workers'],
                                             pin_memory=dl_cfg['pin_memory'])
        print('Dataset Tinyimagenet ready.')
        
    elif ds_cfg['dataset'] == 'imagenet_resize':
        dataset = datasets.ImageFolder(os.path.join(ds_cfg['data_root'], "Imagenet_resize"), 
                                          transform=transform)
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)
        loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=ds_cfg['batch_size'],
                                         shuffle=train,
                                         num_workers=dl_cfg['num_workers'],
                                         pin_memory=dl_cfg['pin_memory'])
        print('Dataset Imagenet resized ready.')
        
    elif ds_cfg['dataset'] == 'lsun_resize':
        dataset = datasets.ImageFolder(os.path.join(ds_cfg['data_root'], "LSUN_resize"), 
                                          transform=transform)
        if 'perturb' in ds_cfg.keys() and ds_cfg['perturb']:
            dataset = PerturbDataset(dataset, train_mode=train)
        loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=ds_cfg['batch_size'],
                                         shuffle=train,
                                         num_workers=dl_cfg['num_workers'],
                                         pin_memory=dl_cfg['pin_memory'])
        print('Dataset LSUN resized ready.')
    else:
        raise NotImplementedError(
            print("Given dataset {} doesn't exist in implemented dataset.".format(dataset))
        )

        
    return loader