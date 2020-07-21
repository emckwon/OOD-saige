import os
import numpy as np
import torch
import argparse
from tensorboardX import SummaryWriter
import shutil
import time
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import sys
sys.path.append('./')
import utils.losses as losses
import utils.metrics as metrics
import utils.optimizer as optim
from models.model_builder import getModel
from datasets.data_loader import getDataLoader
from config import cfg


def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key], summary['epoch'])
        

def train_epoch_wo_outlier(model, optimizer, in_loader, loss_func, cur_epoch, op_cfg, writer):
    model.train()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    #TODO: What would be matter if out_data_size < in_data_size
    for cur_iter, in_set in enumerate(in_loader):
        #TODO: Dimension of in_set and out_set should be checked!
        
        # Data to GPU
        data = in_set[0]
        targets = in_set[1]
        if cur_iter == 0:
            writer.add_image('in_dist target {}'.format(targets[0]), data[0], cur_epoch)
        data, targets = data.cuda(), targets.cuda()
        
        # Adjust Learning rate
        lr = optim.get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
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
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!!
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
    
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'classifier_acc': correct / in_data_size,
        'lr': optim.get_lr_at_epoch(op_cfg, cur_epoch),
        'epoch': cur_epoch,
    }
    
    return summary
  
    
def valid_epoch_wo_outlier(model, in_loader, loss_func, cur_epoch):
    model.eval()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, in_set in enumerate(in_loader):        
        # Data to GPU
        data = in_set[0]
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Foward propagation and Calculate loss
        logits = model(data)
        
        loss = loss_func(logits, targets)
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!!
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
    
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'classifier_acc': correct / in_data_size,
        'epoch': cur_epoch,
    }
    
    return summary
    


def train_epoch_w_outlier(model, optimizer, in_loader, out_loader, loss_func, cur_epoch, op_cfg, writer):
    model.train()
    avg_loss = 0
    correct = 0
    total = 0
    in_data_size = len(in_loader.dataset)
    out_data_size = len(out_loader.dataset)
    out_loader.dataset.offset = np.random.randint(len(out_loader.dataset))
    #TODO: What would be matter if out_data_size < in_data_size
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):
        #TODO: Dimension of in_set and out_set should be checked!
        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        if cur_iter == 0:
            writer.add_image('in_dist target {}'.format(targets[0]), data[0], cur_epoch)
        data, targets = data.cuda(), targets.cuda()
        
        # Adjust Learning rate
        lr = optim.get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
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
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!!
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'lr': optim.get_lr_at_epoch(op_cfg, cur_epoch),
        'epoch': cur_epoch,
    }
    
    return summary
  
    
def valid_epoch_w_outlier(model, in_loader, out_loader, loss_func, cur_epoch):
    model.eval()
    avg_loss = 0
    correct = 0
    total = 0
    in_data_size = len(in_loader.dataset)
    out_data_size = len(out_loader.dataset)
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
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!!
        
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'epoch': cur_epoch,
    }
    
    return summary
    
def main():
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer
    model = getModel(cfg['model'])
    optimizer = optim.getOptimizer(model, cfg['optim'])
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
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'epoch' in checkpoint.keys():
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 1
    cudnn.benchmark = True
    
    # Data Loader
    in_train_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="train")
    in_valid_loader = getDataLoader(ds_cfg=cfg['in_dataset'],
                                    dl_cfg=cfg['dataloader'],
                                    split="valid")
    
    if len(cfg['out_dataset']['targets']) != 0:
        out_train_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                         dl_cfg=cfg['dataloader'],
                                         split="train")
        out_valid_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                         dl_cfg=cfg['dataloader'],
                                         split="valid")
    else:
        out_train_loader = None
        out_valid_loader = None
    
    # Result directory and make tensorboard event file
    exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'])
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)
    shutil.copy('./config.py', os.path.join(exp_dir, "config.py"))
    writer_train = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'train'))
    writer_valid = SummaryWriter(logdir=os.path.join(exp_dir, 'log', 'valid'))
    
    # Stats Meters
    #train_meter = TrainMeter()
    #valid_meter = ValidMeter()
    
    # Loss function
    loss_func = losses.getLoss(cfg['loss'])
    
    print("Start training. Result will be saved in {}".format(exp_dir))
    
    for cur_epoch in range(start_epoch, cfg['max_epoch'] + 1):
        if out_train_loader is not None:
            train_summary = train_epoch_w_outlier(model, optimizer, in_train_loader, out_train_loader, loss_func, cur_epoch, cfg['optim'], writer_train)
        else:
            train_summary = train_epoch_wo_outlier(model, optimizer, in_train_loader, loss_func, cur_epoch, cfg['optim'], writer_train)
        summary_write(summary=train_summary, writer=writer_train)
        print("Training result =======> Epoch [{}]/[{}]\nlr: {} | loss: {} | acc: {}".format(cur_epoch, cfg['max_epoch'], train_summary['lr'], train_summary['avg_loss'], train_summary['classifier_acc']))
        
        
        if cur_epoch % cfg['valid_epoch'] == 0:
            if out_valid_loader is not None:
                valid_summary = valid_epoch_w_outlier(model, in_valid_loader, out_valid_loader, loss_func, cur_epoch)
            else:
                valid_summary = valid_epoch_wo_outlier(model, in_valid_loader, loss_func, cur_epoch)
            summary_write(summary=valid_summary, writer=writer_valid)
            print("Validate result =======> Epoch [{}]/[{}]\nloss: {} | acc: {}".format(cur_epoch, cfg['max_epoch'], valid_summary['avg_loss'], valid_summary['classifier_acc']))
        
        if cur_epoch % cfg['ckpt_epoch'] == 0:
            ckpt_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "ckpt")
            if os.path.exists(ckpt_dir) is False:
                os.makedirs(ckpt_dir)
            model_state = model.module.state_dict() if cfg['ngpu'] > 1 else model.state_dict()
            checkpoint = {
                "epoch": cur_epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
            }
            ckpt_name = "checkpoint_epoch_{}".format(cur_epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name + ".pyth")
            torch.save(checkpoint, ckpt_path)
        

if __name__=="__main__":
    print("Setup Training...")
    main()