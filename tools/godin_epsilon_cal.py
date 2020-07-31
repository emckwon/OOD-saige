import os
import numpy as np
import torch
import argparse
import shutil
import time
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import sys
sys.path.append('./')
import utils.losses as losses
import utils.detectors as detectors
import utils.metrics as metrics
import utils.optimizer as optim
from models.model_builder import getModel
from datasets.data_loader import getDataLoader
from config import cfg
from torch.autograd import Variable


def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key], summary['epoch'])
        
    
def valid_epoch_wo_outlier(model, in_loader, epsilon):
    model.eval()
    score = 0
    
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(in_loader):        
        # Data to GPU
        data = in_set[0]
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        ### Input preprocessing
        inputs = Variable(data, requires_grad=True)
        _, h_logits = model(inputs)
        loss = torch.max(h_logits, dim=1).values.mean()
        loss.backward()        

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        
        tempInputs = torch.add(data.data, gradient, alpha=-epsilon)
        ###
        
        # Foward propagation and Calculate loss
        with torch.no_grad():
            f_logits, h_logits = model(Variable(tempInputs))
        
        score += torch.max(h_logits, dim=1).values.sum().cpu()
        
    
    
    summary = {
        'score': score,
    }
    
    return summary
    


def main():
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer
    model = getModel(cfg['model'])
    start_epoch = 1
    max_epoch = 1
    
    # Load model and optimizer
    if cfg['load_ckpt'] != '':
        checkpoint = torch.load(cfg['load_ckpt'], map_location="cpu")
        model.load_state_dict(checkpoint['model_state'])
        print("load model on '{}' is complete.".format(cfg['load_ckpt']))
    cudnn.benchmark = True
    
    # Data Loader
    in_valid_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="valid")
    

    exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'])
    
    # Result directory and make tensorboard event file
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)    
    
    # Loss function
    loss_func = losses.getLoss(cfg['loss'])
    
    # Outlier detector
    print("=======================IMPORTANT CONFIG=======================")
    print(" Model    : {}\n \
Loss     : {}\n".format(cfg['model']['network_kind'], cfg['loss']['loss']))
    print("========Start epsilon regression for GODIN. Result will be saved in {}".format(exp_dir))
    
    logfile = open(os.path.join(exp_dir, "epsilon.txt"), "w")
    
    epsilon = -0.005
    max_epsilon = 0.1
    step = 0.005
    while epsilon <= max_epsilon:
        valid_summary = valid_epoch_wo_outlier(model, in_valid_loader, epsilon)
        summary_log = "Epsilon [{}] => Score [{}]\n".format(epsilon, valid_summary['score'])
        print(summary_log)
        logfile.write(summary_log)
        epsilon += step
            
    logfile.close()

if __name__=="__main__":
    print("Setup epsilon regression...")
    main()