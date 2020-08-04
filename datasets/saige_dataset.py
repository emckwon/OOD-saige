from PIL import Image
import torch
import os

class SaigeDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, dataset, split, transform, targets):
        """
            data_root(str) : Root directory of datasets (e.g. "/home/sr2/HDD2/Openset/")
            split_root(str) : Root directroy of split file (e.g. "/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/")
            dataset(str) : dataset name
            split(str) : ['train', 'valid', 'test']
            transform(torchvision transform) : image transform
            targets(list of str) : using targets
        """
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.transform = transform
        self.targets = targets
        
        self.data_list = []
        f = open(os.path.join(split_root, dataset, split + ".txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            [target, _] = line[:-1].split("/")
            # Transform target
            if target in targets:
                target = targets.index(target)
                self.data_list.append((target, line[:-1]))
            
    def __getitem__(self, idx):
        (target, fpath) = self.data_list[idx]
        img = Image.open(os.path.join(self.data_root, self.dataset, fpath))
        img = self.transform(img)
        return img, target
       
    def __len__(self):
        return len(self.data_list)
    
    
class SaigeDataset2(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, dataset, split, transform, targets):
        """
            Dataset for Daeduck
        """
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.transform = transform
        self.targets = targets
        self.split = split
        self.data_list = []
        f = open(os.path.join(split_root, dataset, split + ".txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            [target, _] = line[:-1].split("/")
            # Transform target
            if target in targets:
                target = targets.index(target)
                self.data_list.append((target, line[:-1]))
            
    def __getitem__(self, idx):
        (target, fpath) = self.data_list[idx]
        img = Image.open(os.path.join(self.data_root, self.dataset, self.split, fpath))
        img = self.transform(img)
        return img, target
       
    def __len__(self):
        return len(self.data_list)