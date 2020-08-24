import torch
import sys
sys.path.append('./')
from tqdm import tqdm
from numpy import arange

from torchvision import transforms as trn
import os

cfg = dict()


# Training config
cfg['finetuning'] = False
cfg['max_epoch'] = 220
cfg['seed'] = 1
cfg['valid_epoch'] = 1  # Epoch term of validation
cfg['ngpu'] = 1

# Log config
cfg['log_step'] = 100  # Step term of log
cfg['exp_root'] = '/home/sr2/Hyeokjun/OOD-saige/results/'
cfg['exp_dir'] = 'Daeduck_Resnet_Pretrained'
cfg['load_dir'] = 'Daeduck_Resnet_Pretrained'
cfg['ckpt_epoch'] = 5  # Epoch term of saving checkpoint file.

# Load Checkpoint
cfg['load_ckpt'] =os.path.join(cfg['exp_root'],cfg['load_dir'],'ckpt','checkpoint_epoch_200.pyth')

#os.path.join(cfg['exp_root'],cfg['load_dir'],'ckpt','checkpoint_epoch_225.pyth')
cfg['mean']=[0.5103,0.5103,0.6657]
cfg['std']=[0.3176,0.3176,0.4561]
cfg['min']=-1.4595
cfg['max']=1.5419

# cfg['mean']=[0,0,0]
# cfg['std']=[1,1,1]
# cfg['min']=-1.9895
# cfg['max']=2.1265

# DataLoader config
cfg['dataloader'] = dict()
cfg['dataloader']['batch_size'] = 1
cfg['dataloader']['num_workers'] = 8
cfg['dataloader']['pin_memory'] = True

# In-Dataset config
cfg['in_dataset'] = dict()
cfg['in_dataset']['batch_size'] = 128
cfg['in_dataset']['dataset'] = 'Daeduck'
cfg['in_dataset']['targets'] = ['slit','short','open','OK','hole']
cfg['in_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(), trn.Resize(256),trn.RandomCrop(224), trn.ToTensor(),trn.Normalize(cfg['mean'], cfg['std'])])
cfg['in_dataset']['valid_transform'] = trn.Compose([ trn.Resize(256),trn.CenterCrop(224),trn.ToTensor(),trn.Normalize(cfg['mean'],cfg['std'])])
cfg['in_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
cfg['in_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'
# 

# cfg['train_dataset'] = dict()
# cfg['train_dataset']['split'] = 'train'
# cfg['train_dataset']['batch_size'] = 128
# cfg['train_dataset']['dataset'] = 'cifar100'
# # cfg['train_dataset']['targets'] = ['ok','1','2']
# cfg['train_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(), trn.Resize(224), trn.ToTensor(),trn.Normalize(cfg['mean'], cfg['std'])])
# cfg['train_dataset']['valid_transform'] = trn.Compose([trn.Resize(224), trn.ToTensor(),trn.Normalize(cfg['mean'],cfg['std'])])
# cfg['train_dataset']['data_root'] = '/home/sr1/Extra_HDD/Hyeokjun/'
# cfg['train_dataset']['split_root'] = '/home/sr1/Hyeokjun/OOD-saige/datasets/data_split/'

# cfg['in_dataset']['train_transform'] = trn.Compose([trn.RandomHorizontalFlip(), trn.ToTensor(),trn.Normalize(mean,std)])
# cfg['in_dataset']['valid_transform'] = trn.Compose([trn.ToTensor(),trn.Normalize(mean,std)])

# cfg['in_dataset']['data_root'] = '/home/sr1/Extra_HDD/Openset/'
# cfg['in_dataset']['split_root'] = '/home/sr1/Hyeokjun/OOD-saige/datasets/data_split/'

# # Out-Dataset config
cfg['out_dataset'] =dict()
cfg['out_dataset']['batch_size'] =128
cfg['out_dataset']['dataset'] = 'Daeduck'
cfg['out_dataset']['targets'] = ['dent','bump','bonding']
cfg['out_dataset']['train_transform'] = trn.Compose([ trn.Resize(256), trn.CenterCrop(224),trn.ToTensor(),trn.Normalize(cfg['mean'],cfg['std'])])
cfg['out_dataset']['valid_transform'] = trn.Compose([ trn.Resize(256), trn.CenterCrop(224),trn.ToTensor(),trn.Normalize(cfg['mean'],cfg['std'])])
cfg['out_dataset']['data_root'] = '/home/sr2/HDD2/Openset/'
# # # #cfg['out_dataset']['data_root'] = '/home/sr2/HDD/Hyeokjun/'
cfg['out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'


## Public Image Dataset to validate the hyperparameter choice

cfg['public_out_dataset'] =dict()
cfg['public_out_dataset']['batch_size'] =128
cfg['public_out_dataset']['dataset'] = 'imagenet_resize'
cfg['public_out_dataset']['train_transform'] = trn.Compose([ trn.Resize(256),trn.CenterCrop(224),trn.ToTensor(),trn.Normalize(cfg['mean'],cfg['std'])])
cfg['public_out_dataset']['valid_transform'] = trn.Compose([ trn.Resize(256),trn.CenterCrop(224),trn.ToTensor(),trn.Normalize(cfg['mean'],cfg['std'])])
cfg['public_out_dataset']['data_root'] = '/home/sr2/HDD/Hyeokjun/'
# # # #cfg['out_dataset']['data_root'] = '/home/sr2/HDD/Hyeokjun/'
cfg['public_out_dataset']['split_root'] = '/home/sr2/Hyeokjun/OOD-saige/datasets/data_split/'


# # Estimate-Dataset config
# cfg['estimate_dataset'] = dict()
# cfg['estimate_dataset']['batch_size'] = 1
# cfg['estimate_dataset']['dataset'] = 'Daeduck'
# cfg['estimate_dataset']['targets'] = ['slit','short','open','OK','hole','dent','bump','bonding']
# cfg['estimate_dataset']['train_transform'] = trn.Compose([trn.Resize(256),trn.CenterCrop(224),trn.ToTensor(),trn.Normalize(cfg['mean'], cfg['std'])])
# cfg['estimate_dataset']['data_root'] = '/home/sr1/Extra_HDD/Openset/'
# cfg['estimate_dataset']['split_root'] = '/home/sr1/ParkYH/OOD-saige/datasets/data_split/'


# Model config
cfg['model'] = dict()
cfg['model']['network_kind'] = 'resnet34_224'
cfg['model']['net_type']='resnet34'
cfg['model']['depth'] = 40
cfg['model']['widen_factor'] = 2
cfg['model']['num_classes'] = len(cfg['in_dataset']['targets'])
#cfg['model']['num_classes'] = 
cfg['model']['drop_rate'] = 0.3
cfg['model']['image_size']= 224

# Loss config
cfg['loss'] = dict()
cfg['loss']['loss'] = 'cross_entropy_in_distribution'
# cfg['loss']['train_acc']=0.99
# cfg['loss']['lambda_1']=0.07
# cfg['loss']['lambda_2']=0.03


# Detector config
cfg['detector'] = dict()
cfg['detector']['detector'] = 'msp'
cfg['detector']['temperature'] = [1, 20, 40, 80, 100, 150, 200, 500, 800, 900, 1000]
cfg['detector']['epsilon'] = [0.01]  ## 
cfg['detector']['adv_method']= 'FGSM'   # Choose one of these : FGSM, BIM, DeepFool, CWL2
cfg['detector']['adv_noise']= [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  #[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
cfg['detector']['graph']= True

# Optimizer & scheduler config
cfg['optim'] = dict() 
cfg['optim']['max_epoch'] = cfg['max_epoch']
cfg['optim']['optimizer'] = 'sgd'
cfg['optim']['momentum'] = 0.9
cfg['optim']['nesterov'] = True
cfg['optim']['weight_decay'] = 0.0005
cfg['optim']['lr'] = 0.01
cfg['optim']['policy'] = 'cosine'
cfg['optim']['warm_epoch'] = 0 # Warm starting epoch if smaller than zero, no warm starting.
cfg['optim']['warm_lr'] = 0.0 # Warm starting learning rate.


# Additional configuration add here!!

# loader = getDataLoader(ds_cfg=cfg['in_dataset'],
#                                     dl_cfg=cfg['dataloader'],
#                                     split="valid")

# means = []
# stds = []
# mins=[]
# maxs=[]
# count=0
# def meanstd(loader):
#     for img in tqdm(loader):
#     #     print(img)
#         rdata=img[0].data.numpy()[0][0]
#         gdata=img[0].data.numpy()[0][1]
#         bdata=img[0].data.numpy()[0][2]

#         rmean=numpy.mean(rdata)
#         gmean=numpy.mean(gdata)
#         bmean=numpy.mean(bdata)
#         rstd=numpy.std(rdata)
#         gstd=numpy.std(gdata)
#         bstd=numpy.std(bdata)

#         mean=[rmean, gmean, bmean]
#         std=[rstd,gstd,bstd]
#         means.append(mean)
#         stds.append(std)

#     mean = numpy.mean(means,axis=0)
#     std = numpy.mean(stds,axis=0)
#     print(mean, std)
#     return mean, std

# def minmax(loader):
#     for img in tqdm(loader):
#         rmin=numpy.amin(img[0].data.numpy()[0][0])
#         gmin=numpy.amin(img[0].data.numpy()[0][1])
#         bmin=numpy.amin(img[0].data.numpy()[0][2])
#         rmax=numpy.amax(img[0].data.numpy()[0][0])
#         gmax=numpy.amax(img[0].data.numpy()[0][1])
#         bmax=numpy.amax(img[0].data.numpy()[0][2])

#         minpixels=min([rmin,gmin,bmin])
#         maxpixels=max([rmax,gmax,bmax])
        
#         mins.append(minpixels)
#         maxs.append(maxpixels)
        

#     minpixel = min(mins)
#     maxpixel = max(maxs)
#     print(minpixel,maxpixel)
#     return minpixel, maxpixel

# def main():
#     meanstd(loader)
#     minmax(loader)


