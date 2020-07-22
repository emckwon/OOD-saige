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

global global_cfg
global_cfg = dict()


def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key], summary['epoch'])
        
    
def valid_epoch_wo_outlier(model, in_loader, loss_func, cur_epoch):
    global global_cfg  
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
        
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        
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
    

def valid_epoch_w_outlier(model, in_loader, out_loader, loss_func, detector_func, cur_epoch):
    model.eval()
    global global_cfg  
    avg_loss = 0
    correct = 0
    total = 0
    max_iter = 0
    avg_auroc = 0
    avg_aupr = 0
    avg_fpr = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        data, targets = data.cuda(), targets.cuda()
        
        # Foward propagation and Calculate loss and confidence
        logits = model(data)
        
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        confidences_dict = detector_func(logits, targets, global_cfg['detector'])
        confidences = confidences_dict['confidences']
        
        ## METRICS ##
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Calculate OOD metrics (auroc, aupr, fpr)
        (auroc, aupr, fpr) = metrics.get_ood_measures(confidences, targets)
        
        # Add additional metrics!!!
        
        ## Update stats ##
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
        total += targets.size(0)
        max_iter += 1
        avg_auroc += auroc
        avg_aupr += aupr
        avg_fpr += fpr
        
    
    summary = {
        'avg_loss': avg_loss / total,
        'classifier_acc': correct / total,
        'AUROC': avg_auroc / max_iter,
        'AUPR' : avg_aupr / max_iter,
        'FPR95': avg_fpr / max_iter,
        'epoch': cur_epoch,
    }
    
    return summary
    
def main():
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer
    model = getModel(cfg['model'])
    start_epoch = 1
    max_epoch = 5
    
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
    
    if cfg['out_dataset'] is not None:
        out_valid_loader = getDataLoader(ds_cfg=cfg['out_dataset'],
                                         dl_cfg=cfg['dataloader'],
                                         split="valid")
        exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "valid", cfg['out_dataset']['dataset'])
    else:
        out_train_loader = None
        out_valid_loader = None
        exp_dir = os.path.join(cfg['exp_root'], cfg['exp_dir'], "valid", "classifier")
    
    # Result directory and make tensorboard event file
    if os.path.exists(exp_dir) is False:
        os.makedirs(exp_dir)    
    shutil.copy('./config.py', os.path.join(exp_dir, "val_config.py"))
    
    # Loss function
    loss_func = losses.getLoss(cfg['loss'])
    global_cfg['loss'] = cfg['loss']
    
    # Outlier detector
    detector_func = detectors.getDetector(cfg['detector'])  
    global_cfg['detector'] = cfg['detector']
    
    print("Start validation. Result will be saved in {}".format(exp_dir))
    
    logfile = open(os.path.join(exp_dir, "validation_log.txt"), "w")
    for cur_epoch in range(start_epoch, max_epoch + 1):
        if out_valid_loader is not None:
            valid_summary = valid_epoch_w_outlier(model, in_valid_loader,
                                                  out_valid_loader, loss_func,
                                                  detector_func, cur_epoch)
            summary_log = "=========Epoch [{}]/[{}]=========\nloss: {} | acc: {}\nAUROC: {} | AUPR: {} | FPR95: {}\n".format(cur_epoch, max_epoch, valid_summary['avg_loss'], valid_summary['classifier_acc'], valid_summary['AUROC'], valid_summary['AUPR'], valid_summary['FPR95'])
        else:
            valid_summary = valid_epoch_wo_outlier(model, in_valid_loader,
                                                   loss_func, cur_epoch)
            summary_log = "=========Epoch [{}]/[{}]=========\nloss: {} | acc: {}\n".format(cur_epoch, max_epoch, valid_summary['avg_loss'], valid_summary['classifier_acc'])
            
        print(summary_log)
        logfile.write(summary_log)
            
    logfile.close()
        
    

if __name__=="__main__":
    print("Setup validation...")
    main()