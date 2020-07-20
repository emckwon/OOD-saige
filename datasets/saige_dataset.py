from PIL import Image
import torch
import os

__all__ = ['Severstal', 'SDI', 'HBT', 'DAGM']

class Severstal(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, split, transform, target_transform):
        self.data_root = data_root
        self.split_root = split_root
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_list = []
        f = open(os.path.join(split_root, "Severstal", split + ".txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            self.data_list.append(line[:-1])
        
    def __getitem__(self, idx):
        [target, fname] = self.data_list[idx].split("/")
        img = Image.open(os.path.join(self.data_root, "Severstal", target, fname))
        img = self.transform(img)
        target = self.target_transform(target)
        return img, target
       
    def __len__(self):
        return len(self.data_list)
    
    
class DAGM(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, split, transform, target_transform):
        self.data_root = data_root
        self.split_root = split_root
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_list = []
        f = open(os.path.join(split_root, "DAGM", split + ".txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            self.data_list.append(line[:-1])
        
    def __getitem__(self, idx):
        [target, fname] = self.data_list[idx].split("/")
        img = Image.open(os.path.join(self.data_root, "DAGM", target, fname))
        img = self.transform(img)
        target = self.target_transform(target)
        return img, target
       
    def __len__(self):
        return len(self.data_list)
    
    
class HBT(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, split, transform, target_transform):
        self.data_root = data_root
        self.split_root = split_root
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_list = []
        f = open(os.path.join(split_root, "Severstal", split + ".txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            self.data_list.append(line[:-1])
        
    def __getitem__(self, idx):
        [target, fname] = self.data_list[idx].split("/")
        img = Image.open(os.path.join(self.data_root, "Severstal", target, fname))
        img = self.transform(img)
        target = self.target_transform(target)
        return img, target
       
    def __len__(self):
        return len(self.data_list)
    
    
class SDI(torch.utils.data.Dataset):
    def __init__(self, data_root, split_root, split, transform, target_transform):
        self.data_root = data_root
        self.split_root = split_root
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_list = []
        f = open(os.path.join(split_root, "Severstal", split + ".txt"), "r")
        while True:
            line = f.readline()
            if not line: break
            self.data_list.append(line[:-1])
        
    def __getitem__(self, idx):
        [target, fname] = self.data_list[idx].split("/")
        img = Image.open(os.path.join(self.data_root, "Severstal", target, fname))
        img = self.transform(img)
        target = self.target_transform(target)
        return img, target
       
    def __len__(self):
        return len(self.data_list)
        