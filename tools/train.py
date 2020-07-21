import os
import numpy as np
import torch
import argparse
from tensorboardX import SummaryWriter
import shutil
import time
import torch.backends.cudnn as cudnn

import utils.losses as losses # getLoss
import utils.logging as logging # summary_write
import utils.metrics as metrics # topks_correct
import utils.optimizer as optim #getOptimizer
from utils.meters import TrainMeter, ValMeter
from models.model_builder import getModel
from datasets.data_loader import getDataLoader
from config import cfg

#logger = logging.get_logger()

def train_epoch(model, optimizer, in_loader, out_loader, loss_func, cur_epoch, train_meter):
    model.train()
    train_meter.iter_tic()
    in_data_size = len(in_loader)
    out_data_size = len(out_loader)
    out_loader.dataset.offset = np.random.randint(len(out_loader.dataset))
    #TODO: What would be matter if out_data_size < in_data_size
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):
        #TODO: Dimension of in_set and out_set should be checked!
        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Adjust Learning rate
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / in_data_size)
        optim.set_lr(optimizer, lr)
        
        # Foward propagation and Calculate loss
        logits = model(data)
        loss = loss_func(logits, targets)
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        top1_err = [(1.0 - x / targets.size(0)) * 100.0 for x in num_topks_correct]
        
        # Add additional metrics!!!
        
        
        train_meter.iter_toc()
        loss, top1_err = loss.item(), top1_err.item()
        train_meter.update_stats(loss, top1_err, lr)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
    
    summary = train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    
    return summary
  
    
def valid_epoch(model, in_loader, out_loader, loss_func, valid_meter):
    model.eval()
    valid_meter.iter_tic()
    in_data_size = len(in_loader)
    out_data_size = len(out_loader)
    #TODO: What would be matter if out_data_size < in_data_size
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Foward propagation and Calculate loss
        logits = model(data)
        loss = loss_func(logits, targets)
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        top1_err = [(1.0 - x / targets.size(0)) * 100.0 for x in num_topks_correct]
        
        # Add additional metrics!!!
        
        
        valid_meter.iter_toc()
        loss, top1_err = loss.item(), top1_err.item()
        valid_meter.update_stats(loss, top1_err, lr)
        valid_meter.log_iter_stats(cur_epoch, cur_iter)
    
    summary = valid_meter.log_epoch_stats(cur_epoch)
    valid_meter.reset()
    
    return valid
    
def main():
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer
    model = getModel(cfg['model'])
    optimizer = optim.getOptimizer(cfg['optim'])
    start_epoch = 1
    
    # Load model and optimizer
    if cfg['load_ckpt'] != '':
        ckpt_path = os.path.join(cfg['exp_root'], cfg['exp_dir'], "ckpt", cfg['load_ckpt'])
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": sd,
            "optimizer_state": optimizer.state_dict(),
        }
        """
        model.load_state_dict(checkpoint['model_state'])
        if not cfg['finetuning']:
            optimizer_state.load_state_dict(checkpoint['optimizer_state'])
        if 'epoch' in checkpoint.keys()
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 1
    
    # Data Loader
    in_train_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="train")
    in_valid_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg['dataloader'],
                                    split="valid")
    out_train_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                     dl_cfg=cfg['dataloader'],
                                     split="train")
    out_valid_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                     dl_cfg=cfg['dataloader'],
                                     split="valid")
    
    # Result directory and make tensorboard event file
    exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'])
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)
    writer_train = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'train'))
    writer_valid = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'valid'))
    
    # Stats Meters
    train_meter = TrainMeter()
    valid_meter = ValidMeter()
    
    # Loss function
    loss_func = losses.getLoss(cfg['loss'])
    
    for cur_epoch in range(start_epoch, cfg['max_epoch'] + 1):
        train_summary = train_epoch(model, optimizer, in_train_loader, out_train_loader, loss_func, cur_epoch, train_meter)
        logging.summary_write(summary=train_summary, writer=writer_train)
        
        if cur_epoch % cfg['valid_epoch'] == 0:
            valid_result = valid_epoch(model, in_valid_loader, out_valid_loader, loss_func, valid_meter)
            logging.summary_write(summary=valid_summary, writer=writer_valid)
        
        if cur_epoch % cfg['ckpt_epoch'] == 0:
            ckpt_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "ckpt")
            if os.path.exist(ckpt_dir) is False:
                os.makedirs(ckpt_dir)
            model_state = model.module.state_dict() if cfg['ngpu'] > 1 else model.state_dict()
            checkpoint = {
                "epoch": cur_epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
            }
            ckpt_name = "checkpoint_epoch_{}".format(cur_epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        

if __name__=="__main__":
    print("Setup Training...")
    main()