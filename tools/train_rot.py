import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tensorboardX import SummaryWriter
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
from utils.pgd_attack import RotPGDAttack
from config import cfg

global global_cfg
global_cfg = dict()


def summary_write(summary, writer):
    for key in summary.keys():
        writer.add_scalar(key, summary[key], summary['epoch'])
        

def train_epoch_wo_outlier(model, optimizer, in_loader, attack_in, cur_epoch, op_cfg, writer):
    global global_cfg
    model.train()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, (x_tf_0, x_tf_90, x_tf_180, x_tf_270, targets) in enumerate(in_loader):
        
        batch_size = x_tf_0.shape[0]
        
        assert x_tf_0.shape[0] == \
            x_tf_90.shape[0] == \
            x_tf_180.shape[0] == \
            x_tf_270.shape[0] == \
            targets.shape[0]
            #x_tf_trans.shape[0] == \
            #target_trans_x.shape[0] == \
            #target_trans_y.shape[0] == \
            
        batch = np.concatenate((
            x_tf_0,
            x_tf_90,
            x_tf_180,
            x_tf_270
        ), 0)
        batch = torch.FloatTensor(batch).cuda()
        
        target_rots = torch.cat((
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size)
        ), 0).long()
        
        if attack_in is not None:
            # Process PGD attack
            batch = attack_in.perturb(batch, batch_size, torch.cat((targets, target_rots), 0).cuda())
            batch = batch.cuda()
        
        if cur_iter == 0:
            writer.add_image('Original', batch[0], cur_epoch)
            writer.add_image('Rot90', batch[batch_size], cur_epoch)
            writer.add_image('Rot180', batch[batch_size * 2], cur_epoch)
            writer.add_image('Rot270', batch[batch_size * 3], cur_epoch)
        
         # Adjust Learning rate
        lr = optim.get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
        optim.set_lr(optimizer, lr)
        
        logits, pen = model(batch)
        
        classification_logits = logits[:batch_size]
        rot_logits            = model.rot_head(pen[:4*batch_size])
        #x_trans_logits        = model.x_trans_head(pen[4*batch_size:])
        #y_trans_logits        = model.y_trans_head(pen[4*batch_size:])
        
        
        classification_loss = F.cross_entropy(classification_logits, targets.cuda())
        rot_loss = F.cross_entropy(rot_logits, target_rots.cuda()) * global_cfg['loss']['rot_weight']
#         x_trans_loss = F.cross_entropy(x_trans_logits, target_trans_x.cuda()) * global_cfg['loss']['trans_weight']
#         y_trans_loss = F.cross_entropy(y_trans_logits, target_trans_y.cuda()) * global_cfg['loss']['trans_weight']
        
        
        #loss = classification_loss + ((rot_loss + x_trans_loss + y_trans_loss) / 3.0)
        loss = classification_loss + rot_loss
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:batch_size], targets.cuda(), (1,))
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
  
    
def valid_epoch_wo_outlier(model, in_loader, cur_epoch):
    global global_cfg
    model.eval()
    avg_loss = 0
    correct = 0
    in_data_size = len(in_loader.dataset)
    for cur_iter, (x_tf_0, x_tf_90, x_tf_180, x_tf_270, targets) in enumerate(in_loader):
        
        batch_size = x_tf_0.shape[0]
        
        assert x_tf_0.shape[0] == \
            x_tf_90.shape[0] == \
            x_tf_180.shape[0] == \
            x_tf_270.shape[0] == \
            targets.shape[0]
#             x_tf_trans.shape[0] == \
#             target_trans_x.shape[0] == \
#             target_trans_y.shape[0] == \
            
        
        batch = np.concatenate((
            x_tf_0,
            x_tf_90,
            x_tf_180,
            x_tf_270
            #x_tf_trans
        ), 0)
        batch = torch.FloatTensor(batch).cuda()
        
        target_rots = torch.cat((
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size)
        ), 0).long()      
        
        logits, pen = model(batch)
        
        classification_logits = logits[:batch_size]
        rot_logits            = model.rot_head(pen[:4*batch_size])
#         x_trans_logits        = net.x_trans_head(pen[4*batch_size:])
#         y_trans_logits        = net.y_trans_head(pen[4*batch_size:])
        
        classification_loss = F.cross_entropy(classification_logits, targets.cuda())
        rot_loss = F.cross_entropy(rot_logits, target_rots.cuda()) * global_cfg['loss']['rot_weight']
#         x_trans_loss = F.cross_entropy(x_trans_logits, target_trans_x.cuda()) * global_cfg['loss']['trans_weight']
#         y_trans_loss = F.cross_entropy(y_trans_logits, target_trans_y.cuda()) * global_cfg['loss']['trans_weight']
        
        #loss = classification_loss + ((rot_loss + x_trans_loss + y_trans_loss) / 3.0)
        loss = classification_loss + rot_loss
        
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:batch_size], targets.cuda(), (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Add additional metrics!!      
        loss, top1_correct = loss.item(), top1_correct.item()
        avg_loss += loss
        correct += top1_correct
    
    summary = {
        'avg_loss': avg_loss / in_data_size,
        'classifier_acc': correct / in_data_size,
        'epoch': cur_epoch,
    }
    
    return summary
    


def train_epoch_w_outlier(model, optimizer, in_loader, out_loader, loss_func, detector_func, cur_epoch, op_cfg, writer):
    global global_cfg
    model.train()
    avg_loss = 0
    correct = 0
    total = 0
    in_data_size = len(in_loader.dataset)
    out_loader.dataset.offset = np.random.randint(len(out_loader.dataset))
    for cur_iter, (in_set, out_set) in enumerate(zip(in_loader, out_loader)):
        #TODO: Dimension of in_set and out_set should be checked!
        
        # Data to GPU
        data = torch.cat((in_set[0], out_set[0]), 0)
        targets = in_set[1]
        if cur_iter == 0:
            writer.add_image('in_dist sample, target:[{}]'.format(targets[0]), in_set[0][0], cur_epoch)
            writer.add_image('out_dist sample', out_set[0][0], cur_epoch)
        data, targets = data.cuda(), targets.cuda()
        
        # Adjust Learning rate
        lr = optim.get_lr_at_epoch(op_cfg, cur_epoch + float(cur_iter) / in_data_size)
        optim.set_lr(optimizer, lr)
        
        # Foward propagation and Calculate loss and confidence
        logits = model(data)
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        global_cfg['detector']['model'] = model
        global_cfg['detector']['data'] = data
        loss_dict = loss_func(logits, targets, global_cfg['loss'])
        loss = loss_dict['loss']
        confidences_dict = detector_func(logits, targets, global_cfg['detector'])
        confidences = confidences_dict['confidences']

        
        # Back propagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## METRICS ##
        # Calculate classifier error about in-distribution sample
        num_topks_correct = metrics.topks_correct(logits[:len(targets)], targets, (1,))
        [top1_correct] = [x for x in num_topks_correct]
        
        # Calculate OOD metrics (auroc, aupr, fpr)
        #(auroc, aupr, fpr) = metrics.get_ood_measures(confidences, targets)
        
        # Add additional metrics!!!
        
        
        ## UDATE STATS ##
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
  
    
def valid_epoch_w_outlier(model, in_loader, out_loader, loss_func, detector_func, cur_epoch):
    global global_cfg  
    model.eval()
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
        global_cfg['loss']['model'] = model
        global_cfg['loss']['data'] = data
        global_cfg['detector']['model'] = model
        global_cfg['detector']['data'] = data
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
    global global_cfg
    # Reproducibility
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    # Model & Optimizer
    model = getModel(cfg['model'])
    model.rot_head = nn.Linear(model.nChannels, 4)
    model.rot_head.cuda()
    optimizer = optim.getOptimizer(model, cfg['optim'])
    start_epoch = 1
    
    # Load model and optimizer
    if cfg['load_ckpt'] != '':
        checkpoint = torch.load(cfg['load_ckpt'], map_location="cpu")
        model.load_state_dict(checkpoint['model_state'])
        print("load model on '{}' is complete.".format(cfg['load_ckpt']))
        if not cfg['finetuning']:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'epoch' in checkpoint.keys() and not cfg['finetuning']:
            start_epoch = checkpoint['epoch']
            print("Restore epoch {}".format(start_epoch))
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
    attack_in = None
    if 'PGD' in cfg.keys() and cfg['PGD'] is not None:
        attack_in = RotPGDAttack(model=model, eps=cfg['PGD']['epsilon'],
                                  nb_iter=cfg['PGD']['iters'],
                              eps_iter=cfg['PGD']['iter_size'], rand_init=True,
                                  loss_func='CE')
    
    if cfg['out_dataset'] is not None:
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
    global_cfg['loss'] = cfg['loss']
    
    # Outlier detector
    detector_func = detectors.getDetector(cfg['detector'])
    global_cfg['detector'] = cfg['detector']
    
    print("=======================IMPORTANT CONFIG=======================")
    print(" Model    : {}\n \
Loss     : {}\n \
Detector : {}\n \
Optimizer: {}\n".format(cfg['model']['network_kind'], cfg['loss']['loss'], cfg['detector']['detector'], cfg['optim']['optimizer']))
    print("============Start training. Result will be saved in {}".format(exp_dir))
    
    for cur_epoch in range(start_epoch, cfg['max_epoch'] + 1):
        if out_train_loader is not None:
            train_summary = train_epoch_w_outlier(model, optimizer, in_train_loader, out_train_loader, loss_func, detector_func, cur_epoch, cfg['optim'], writer_train)
        else:
            train_summary = train_epoch_wo_outlier(model, optimizer, in_train_loader, attack_in, cur_epoch, cfg['optim'], writer_train)
        summary_write(summary=train_summary, writer=writer_train)
        print("Training result=========Epoch [{}]/[{}]=========\nlr: {} | loss: {} | acc: {}".format(cur_epoch, cfg['max_epoch'], train_summary['lr'], train_summary['avg_loss'], train_summary['classifier_acc']))
        
        
        if cur_epoch % cfg['valid_epoch'] == 0:
            if out_valid_loader is not None:
                valid_summary = valid_epoch_w_outlier(model, in_valid_loader, out_valid_loader, loss_func, detector_func, cur_epoch)
            else:
                valid_summary = valid_epoch_wo_outlier(model, in_valid_loader, cur_epoch)
            summary_write(summary=valid_summary, writer=writer_valid)
            print("Validate result=========Epoch [{}]/[{}]=========\nloss: {} | acc: {}".format(cur_epoch, cfg['max_epoch'], valid_summary['avg_loss'], valid_summary['classifier_acc']))
        
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